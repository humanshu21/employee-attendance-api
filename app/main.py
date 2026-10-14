"""Employee Attendance & Analytics API.

Run from the repository root with uvicorn app.main:app --port 8000.
MongoDB connection settings come from MONGO_URI and MONGO_DB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError


load_dotenv(override=False)

logger = logging.getLogger(__name__)
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000
PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
EpochMillis = Annotated[
    StrictInt,
    Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS),
]

client = MongoClient(
    os.getenv("MONGO_URI") or "mongodb://localhost:27017",
    tz_aware=True,
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=5_000,
)
db = client[os.getenv("MONGO_DB") or "attendance_db"]


def create_indexes() -> None:
    """Create indexes used for natural-key writes, filters, lookups and sorts."""
    db.employees.create_index(
        [("emp_code", ASCENDING)], unique=True, name="uq_employees_emp_code"
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="ix_employees_department_joined_on",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="ix_employees_department_emp_code",
    )
    db.employees.create_index(
        [("joined_on", ASCENDING)], name="ix_employees_joined_on"
    )

    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_attendance_emp_code_date",
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="ix_attendance_date_emp_code",
    )
    db.attendance_logs.create_index(
        [
            ("status", ASCENDING),
            ("date", DESCENDING),
            ("emp_code", ASCENDING),
        ],
        name="ix_attendance_status_date_emp_code",
    )
    db.attendance_logs.create_index(
        [
            ("emp_code", ASCENDING),
            ("punch_in", DESCENDING),
            ("date", DESCENDING),
        ],
        name="ix_attendance_emp_punch_in",
    )
    db.attendance_logs.create_index(
        [
            ("date", ASCENDING),
            ("late_minutes", ASCENDING),
            ("emp_code", ASCENDING),
        ],
        name="ix_attendance_date_late_emp_code",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        create_indexes()
    except PyMongoError:
        # Keep the process alive so /health can report a database outage as 503.
        logger.exception("MongoDB indexes could not be created during startup")
    yield
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmployeeCreate(StrictBody):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(
        max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(
        default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    shift_end: str = Field(
        default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    joined_on: date

    @model_validator(mode="after")
    def distinct_shift_times(self) -> EmployeeCreate:
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInRequest(StrictBody):
    emp_code: str
    punched_at: EpochMillis | None = None
    status: PresenceStatus = "PRESENT"

    @model_validator(mode="before")
    @classmethod
    def reject_null_timestamp(cls, values: object) -> object:
        if isinstance(values, dict) and "punched_at" in values and values["punched_at"] is None:
            raise ValueError("punched_at may be omitted but cannot be null")
        return values


class PunchOutRequest(StrictBody):
    emp_code: str
    punched_at: EpochMillis | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_timestamp(cls, values: object) -> object:
        if isinstance(values, dict) and "punched_at" in values and values["punched_at"] is None:
            raise ValueError("punched_at may be omitted but cannot be null")
        return values


class RegularizeRequest(StrictBody):
    status: Status | None = None
    punch_in: EpochMillis | None = None
    punch_out: EpochMillis | None = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def reject_null_optional_fields(cls, values: object) -> object:
        if isinstance(values, dict):
            for name in ("status", "punch_in", "punch_out"):
                if name in values and values[name] is None:
                    raise ValueError(f"{name} may be omitted but cannot be null")
        return values

    @model_validator(mode="after")
    def require_a_regularized_field(self) -> RegularizeRequest:
        if self.status is None and self.punch_in is None and self.punch_out is None:
            raise ValueError(
                "at least one of status, punch_in, or punch_out is required"
            )
        return self


# --------------------------------------------------------------------------- #
# Time and serialization helpers
# --------------------------------------------------------------------------- #
def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        # PyMongo can be configured to return naive UTC datetimes; treat them as UTC.
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def truncate_to_second(value: datetime) -> datetime:
    return as_utc(value).replace(microsecond=0)


def now_utc_millis() -> datetime:
    current = datetime.now(UTC)
    return current.replace(microsecond=(current.microsecond // 1000) * 1000)


def epoch_ms_to_datetime(value: int) -> datetime:
    instant = EPOCH + timedelta(milliseconds=value)
    return truncate_to_second(instant)


def datetime_to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    delta = as_utc(value) - EPOCH
    return (
        delta.days * 86_400_000
        + delta.seconds * 1_000
        + delta.microseconds // 1_000
    )


def parse_shift_time(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour=hour, minute=minute)


def is_overnight(employee: dict) -> bool:
    return employee["shift_end"] <= employee["shift_start"]


def attendance_date(punch_in: datetime, employee: dict) -> date:
    local = truncate_to_second(punch_in).astimezone(IST)
    result = local.date()
    if is_overnight(employee):
        end = parse_shift_time(employee["shift_end"])
        if local.time().replace(tzinfo=None) < end:
            result -= timedelta(days=1)
    return result


def compute_late_minutes(
    punch_in: datetime, shift_start: str, attendance_day: date
) -> int:
    local_punch = truncate_to_second(punch_in).astimezone(IST)
    shift_start_at = datetime.combine(
        attendance_day, parse_shift_time(shift_start), tzinfo=IST
    )
    elapsed_seconds = int((local_punch - shift_start_at).total_seconds())
    if elapsed_seconds <= 10 * 60:
        return 0
    return max(0, elapsed_seconds // 60)


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    duration = truncate_to_second(punch_out) - truncate_to_second(punch_in)
    seconds = duration.days * 86_400 + duration.seconds
    value = (Decimal(seconds) / Decimal(3_600)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return float(value)


def compute_overtime_minutes(
    punch_out: datetime,
    shift_start: str,
    shift_end: str,
    attendance_day_value: str,
) -> int:
    shift_end_day = date.fromisoformat(attendance_day_value)
    if shift_end <= shift_start:
        shift_end_day += timedelta(days=1)
    shift_end_at = datetime.combine(
        shift_end_day, parse_shift_time(shift_end), tzinfo=IST
    )
    local_punch_out = truncate_to_second(punch_out).astimezone(IST)
    elapsed_seconds = int((local_punch_out - shift_end_at).total_seconds())
    if elapsed_seconds < 30 * 60:
        return 0
    return max(0, elapsed_seconds // 60)


def derive_attendance_values(
    employee: dict,
    status: Status,
    attendance_day_value: str,
    punch_in: datetime | None,
    punch_out: datetime | None,
) -> dict:
    if status not in PRESENCE_STATUSES:
        return {
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
        }

    if punch_in is None:
        raise HTTPException(422, "presence status requires punch_in")

    punch_in = truncate_to_second(punch_in)
    if punch_out is not None:
        punch_out = truncate_to_second(punch_out)

    day = date.fromisoformat(attendance_day_value)
    late_minutes = compute_late_minutes(
        punch_in, employee["shift_start"], day
    )
    if punch_out is None:
        return {
            "work_hours": None,
            "late_minutes": late_minutes,
            "overtime_minutes": 0,
            "half_day": False,
        }

    work_hours = compute_work_hours(punch_in, punch_out)
    return {
        "work_hours": work_hours,
        "late_minutes": late_minutes,
        "overtime_minutes": compute_overtime_minutes(
            punch_out,
            employee["shift_start"],
            employee["shift_end"],
            attendance_day_value,
        ),
        "half_day": Decimal(str(work_hours)) < Decimal("4.50"),
    }


def employee_response(document: dict) -> dict:
    return {
        "emp_code": document["emp_code"],
        "name": document["name"],
        "email": document["email"],
        "department": document["department"],
        "shift_start": document["shift_start"],
        "shift_end": document["shift_end"],
        "joined_on": document["joined_on"],
        "created_at": datetime_to_epoch_ms(document["created_at"]),
    }


def _history_entry_response(entry: dict) -> dict:
    changes = {}
    for field_name, pair in (entry.get("changes") or {}).items():
        normalized = {}
        for direction in ("from", "to"):
            value = pair.get(direction)
            if field_name in ("punch_in", "punch_out"):
                value = datetime_to_epoch_ms(value)
            normalized[direction] = value
        changes[field_name] = normalized
    return {
        "at": datetime_to_epoch_ms(entry["at"]),
        "by": entry["by"],
        "reason": entry["reason"],
        "changes": changes,
    }


def attendance_response(document: dict) -> dict:
    return {
        "emp_code": document["emp_code"],
        "date": document["date"],
        "status": document["status"],
        "punch_in": (
            datetime_to_epoch_ms(truncate_to_second(document["punch_in"]))
            if document.get("punch_in") is not None
            else None
        ),
        "punch_out": (
            datetime_to_epoch_ms(truncate_to_second(document["punch_out"]))
            if document.get("punch_out") is not None
            else None
        ),
        "work_hours": document.get("work_hours"),
        "late_minutes": document.get("late_minutes", 0),
        "overtime_minutes": document.get("overtime_minutes", 0),
        "half_day": document.get("half_day", False),
        "history": [
            _history_entry_response(entry)
            for entry in (document.get("history") or [])
        ],
    }


def parse_month_bounds(month: str) -> tuple[str, str, date, date]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise HTTPException(422, "month must use YYYY-MM")
    try:
        year, month_number = (int(part) for part in month.split("-"))
        first = date(year, month_number, 1)
    except ValueError as exc:
        raise HTTPException(422, "month must be a valid YYYY-MM month") from exc
    if month_number == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month_number + 1, 1)
    last = next_month - timedelta(days=1)
    return first.isoformat(), next_month.isoformat(), first, last


def round_half_up_expression(value: object, decimal_places: int) -> dict:
    scale = 10**decimal_places
    return {
        "$divide": [
            {"$floor": {"$add": [{"$multiply": [value, scale]}, 0.5]}},
            scale,
        ]
    }


def _weekday_expression(date_expression: object) -> dict:
    day_of_week = {"$dayOfWeek": date_expression}
    return {"$in": [day_of_week, [2, 3, 4, 5, 6]]}


def _calendar_day_from_string(value_expression: object) -> dict:
    return {"$dateFromString": {"dateString": value_expression, "timezone": "UTC"}}


# --------------------------------------------------------------------------- #
# Aggregation builders: analytics and /admin/explain share these pipelines.
# --------------------------------------------------------------------------- #
def build_employee_monthly_pipeline(emp_code: str, month: str) -> list[dict]:
    month_start, month_end_exclusive, first, last = parse_month_bounds(month)
    first_expr = _calendar_day_from_string({"$literal": first.isoformat()})
    last_expr = _calendar_day_from_string({"$literal": last.isoformat()})
    next_day_after_end = {
        "$dateAdd": {"startDate": last_expr, "unit": "day", "amount": 1}
    }
    day_count = {
        "$dateDiff": {
            "startDate": "$_working_start",
            "endDate": next_day_after_end,
            "unit": "day",
        }
    }
    weekdays = {
        "$map": {
            "input": {"$range": [0, day_count]},
            "as": "offset",
            "in": {
                "$dayOfWeek": {
                    "$dateAdd": {
                        "startDate": "$_working_start",
                        "unit": "day",
                        "amount": "$$offset",
                    }
                }
            },
        }
    }
    return [
        {"$match": {"emp_code": emp_code}},
        {
            "$set": {
                "_month_start": first_expr,
                "_month_end": last_expr,
                "_joined": _calendar_day_from_string("$joined_on"),
            }
        },
        {
            "$set": {
                "_working_start": {
                    "$cond": [
                        {"$gt": ["$_joined", "$_month_start"]},
                        "$_joined",
                        "$_month_start",
                    ]
                }
            }
        },
        {
            "$set": {
                "working_days": {
                    "$cond": [
                        {"$gt": ["$_joined", "$_month_end"]},
                        0,
                        {
                            "$size": {
                                "$filter": {
                                    "input": weekdays,
                                    "as": "weekday",
                                    "cond": {
                                        "$and": [
                                            {"$gte": ["$$weekday", 2]},
                                            {"$lte": ["$$weekday", 6]},
                                        ]
                                    },
                                }
                            }
                        },
                    ]
                }
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"employee_code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$employee_code"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lt": ["$date", month_end_exclusive]},
                                ]
                            }
                        }
                    }
                ],
                "as": "_logs",
            }
        },
        {
            "$unwind": {
                "path": "$_logs",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$set": {
                "_log_weekday": {
                    "$cond": [
                        {"$ne": [{"$ifNull": ["$_logs.date", None]}, None]},
                        _weekday_expression(
                            _calendar_day_from_string("$_logs.date")
                        ),
                        False,
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "working_days": {"$first": "$working_days"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$_logs.status", list(PRESENCE_STATUSES)]},
                                    "$_log_weekday",
                                ]
                            },
                            {
                                "$cond": [
                                    {"$ifNull": ["$_logs.half_day", False]},
                                    0.5,
                                    1,
                                ]
                            },
                            0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$_logs.status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$_logs.late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$_logs.late_minutes", 0]}
                },
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$_logs.overtime_minutes", 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "emp_code": "$_id",
                "month": {"$literal": month},
                "working_days": 1,
                "present_days": round_half_up_expression("$present_days", 2),
                "leave_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "total_overtime_minutes": 1,
                "attendance_pct": {
                    "$cond": [
                        {"$gt": ["$working_days", 0]},
                        round_half_up_expression(
                            {
                                "$multiply": [
                                    {"$divide": ["$present_days", "$working_days"]},
                                    100,
                                ]
                            },
                            2,
                        ),
                        None,
                    ]
                },
            }
        },
    ]


def build_department_summary_pipeline(
    month: str, department: str | None = None
) -> list[dict]:
    month_start, month_end_exclusive, _, last = parse_month_bounds(month)
    employee_match: dict = {"joined_on": {"$lte": last.isoformat()}}
    if department is not None:
        employee_match["department"] = department

    return [
        {"$match": employee_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"employee_code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$employee_code"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lt": ["$date", month_end_exclusive]},
                                ]
                            }
                        }
                    }
                ],
                "as": "_logs",
            }
        },
        {
            "$unwind": {
                "path": "$_logs",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$set": {
                "_log_weekday": {
                    "$cond": [
                        {"$ne": [{"$ifNull": ["$_logs.date", None]}, None]},
                        _weekday_expression(
                            _calendar_day_from_string("$_logs.date")
                        ),
                        False,
                    ]
                },
                "_has_work_hours": {
                    "$and": [
                        {"$in": ["$_logs.status", list(PRESENCE_STATUSES)]},
                        {"$ne": [{"$ifNull": ["$_logs.work_hours", None]}, None]},
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": {
                    "department": "$department",
                    "emp_code": "$emp_code",
                },
                "headcount": {"$first": 1},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$_logs.status", list(PRESENCE_STATUSES)]},
                                    "$_log_weekday",
                                ]
                            },
                            {
                                "$cond": [
                                    {"$ifNull": ["$_logs.half_day", False]},
                                    0.5,
                                    1,
                                ]
                            },
                            0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": ["$_has_work_hours", "$_logs.work_hours", 0]
                    }
                },
                "work_hours_count": {
                    "$sum": {"$cond": ["$_has_work_hours", 1, 0]}
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$_logs.late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$_logs.late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$_logs.status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$_logs.status", "ON_DUTY"]}, 1, 0]
                    }
                },
            }
        },
        {
            "$group": {
                "_id": "$_id.department",
                "headcount": {"$sum": "$headcount"},
                "present_days": {"$sum": "$present_days"},
                "work_hours_sum": {"$sum": "$work_hours_sum"},
                "work_hours_count": {"$sum": "$work_hours_count"},
                "late_count": {"$sum": "$late_count"},
                "total_late_minutes": {"$sum": "$total_late_minutes"},
                "leave_count": {"$sum": "$leave_count"},
                "on_duty_count": {"$sum": "$on_duty_count"},
            }
        },
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": round_half_up_expression("$present_days", 2),
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        round_half_up_expression(
                            {
                                "$divide": [
                                    "$work_hours_sum",
                                    "$work_hours_count",
                                ]
                            },
                            2,
                        ),
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
    ]


def build_late_leaderboard_pipeline(
    month: str, limit: int, department: str | None = None
) -> list[dict]:
    month_start, month_end_exclusive, _, _ = parse_month_bounds(month)
    pipeline = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lt": month_end_exclusive},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "_employee",
            }
        },
        {"$unwind": "$_employee"},
    ]
    if department is not None:
        pipeline.append({"$match": {"_employee.department": department}})
    pipeline.extend(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$_employee.name",
                    "department": "$_employee.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def build_department_trend_pipeline(
    department: str, from_day: date, to_day: date
) -> list[dict]:
    start_expr = _calendar_day_from_string({"$literal": from_day.isoformat()})
    range_length = {
        "$add": [
            {
                "$dateDiff": {
                    "startDate": start_expr,
                    "endDate": _calendar_day_from_string(
                        {"$literal": to_day.isoformat()}
                    ),
                    "unit": "day",
                }
            },
            1,
        ]
    }
    return [
        {"$match": {"department": department}},
        {"$limit": 1},
        {
            "$set": {
                "_days": {
                    "$map": {
                        "input": {"$range": [0, range_length]},
                        "as": "offset",
                        "in": {
                            "$dateAdd": {
                                "startDate": start_expr,
                                "unit": "day",
                                "amount": "$$offset",
                            }
                        },
                    }
                }
            }
        },
        {"$unwind": "$_days"},
        {
            "$set": {
                "_date": {
                    "$dateToString": {
                        "date": "$_days",
                        "format": "%Y-%m-%d",
                        "timezone": "UTC",
                    }
                }
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"dept": "$department", "day": "$_date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", "$$dept"]},
                                    {"$lte": ["$joined_on", "$$day"]},
                                ]
                            }
                        }
                    },
                    {"$count": "value"},
                ],
                "as": "_headcount",
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"dept": "$department", "day": "$_date"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$date", "$$day"]}}},
                    {
                        "$lookup": {
                            "from": "employees",
                            "localField": "emp_code",
                            "foreignField": "emp_code",
                            "let": {"dept": "$$dept", "day": "$$day"},
                            "pipeline": [
                                {
                                    "$match": {
                                        "$expr": {
                                            "$and": [
                                                {"$eq": ["$department", "$$dept"]},
                                                {"$lte": ["$joined_on", "$$day"]},
                                            ]
                                        }
                                    }
                                }
                            ],
                            "as": "_employee",
                        }
                    },
                    {"$unwind": "$_employee"},
                    {
                        "$group": {
                            "_id": None,
                            "present_count": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$in": [
                                                "$status",
                                                list(PRESENCE_STATUSES),
                                            ]
                                        },
                                        {
                                            "$cond": [
                                                {"$ifNull": ["$half_day", False]},
                                                0.5,
                                                1,
                                            ]
                                        },
                                        0,
                                    ]
                                }
                            },
                            "late_count": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$gt": [
                                                {"$ifNull": ["$late_minutes", 0]},
                                                0,
                                            ]
                                        },
                                        1,
                                        0,
                                    ]
                                }
                            },
                        }
                    },
                ],
                "as": "_daily_logs",
            }
        },
        {
            "$set": {
                "headcount": {
                    "$ifNull": [{"$arrayElemAt": ["$_headcount.value", 0]}, 0]
                },
                "present_count": {
                    "$ifNull": [
                        {"$arrayElemAt": ["$_daily_logs.present_count", 0]},
                        0,
                    ]
                },
                "late_count": {
                    "$ifNull": [
                        {"$arrayElemAt": ["$_daily_logs.late_count", 0]},
                        0,
                    ]
                },
                "is_working_day": _weekday_expression("$_days"),
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        round_half_up_expression(
                            {"$divide": ["$present_count", "$headcount"]}, 4
                        ),
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"_days": 1},
                "output": {
                    "_moving_avg_raw": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {"$sort": {"_days": 1}},
        {
            "$project": {
                "_id": 0,
                "department": 1,
                "date": "$_date",
                "is_working_day": 1,
                "headcount": 1,
                "present_count": round_half_up_expression("$present_count", 2),
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$_moving_avg_raw", None]},
                        None,
                        round_half_up_expression("$_moving_avg_raw", 4),
                    ]
                },
            }
        },
    ]


def build_attendance_query(
    emp_code: str | None,
    date_from: date | None,
    date_to: date | None,
    status: Status | None,
) -> dict:
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    date_query = {}
    if date_from is not None:
        date_query["$gte"] = date_from.isoformat()
    if date_to is not None:
        date_query["$lte"] = date_to.isoformat()
    if date_query:
        query["date"] = date_query
    if status is not None:
        query["status"] = status
    return query


# --------------------------------------------------------------------------- #
# HTTP endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    try:
        db.command("ping")
    except PyMongoError as exc:
        raise HTTPException(503, "MongoDB is unavailable") from exc
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate) -> dict:
    document = body.model_dump()
    document["joined_on"] = body.joined_on.isoformat()
    document["created_at"] = now_utc_millis()
    try:
        db.employees.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(409, "emp_code already exists") from exc
    return employee_response(document)


@app.get("/employees")
def list_employees(
    department: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    query = {"department": department} if department is not None else {}
    skip = (page - 1) * page_size
    total = db.employees.count_documents(query)
    documents = (
        db.employees.find(query, {"_id": 0})
        .sort([("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [employee_response(document) for document in documents],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest) -> dict:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    punch_in_at = (
        epoch_ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else truncate_to_second(datetime.now(UTC))
    )
    day = attendance_date(punch_in_at, employee).isoformat()
    document = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": punch_in_at,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(
            punch_in_at, employee["shift_start"], date.fromisoformat(day)
        ),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(document)
    except DuplicateKeyError as exc:
        # The unique (emp_code, date) index arbitrates simultaneous punch-ins.
        raise HTTPException(
            409, "attendance record already exists for this date"
        ) from exc
    return attendance_response(document)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest) -> dict:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    punch_out_at = (
        epoch_ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else truncate_to_second(datetime.now(UTC))
    )
    candidate = db.attendance_logs.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$lte": punch_out_at},
        },
        sort=[("punch_in", DESCENDING), ("date", DESCENDING)],
    )
    if candidate is None:
        raise HTTPException(404, "no punch-in found")
    if candidate.get("punch_out") is not None:
        raise HTTPException(409, "record is already punched out")

    punch_in_at = truncate_to_second(candidate["punch_in"])
    duration = punch_out_at - punch_in_at
    if duration.total_seconds() <= 0 or duration > timedelta(hours=24):
        raise HTTPException(
            422, "punched_at must be after punch_in and within 24 hours"
        )

    derived = derive_attendance_values(
        employee,
        candidate["status"],
        candidate["date"],
        punch_in_at,
        punch_out_at,
    )
    result = db.attendance_logs.update_one(
        {
            "emp_code": body.emp_code,
            "date": candidate["date"],
            "punch_in": candidate["punch_in"],
            "punch_out": None,
        },
        {"$set": {"punch_in": punch_in_at, "punch_out": punch_out_at, **derived}},
    )
    if result.modified_count != 1:
        raise HTTPException(409, "record was already punched out")
    updated = {
        **candidate,
        "punch_in": punch_in_at,
        "punch_out": punch_out_at,
        **derived,
    }
    return attendance_response(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    status: Status | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    query = build_attendance_query(emp_code, date_from, date_to, status)
    skip = (page - 1) * page_size
    total = db.attendance_logs.count_documents(query)
    documents = (
        db.attendance_logs.find(query, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [attendance_response(document) for document in documents],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def _snapshot_filter(document: dict) -> dict:
    # Compare all mutable fields to prevent stale concurrent overwrites.
    fields = (
        "status",
        "punch_in",
        "punch_out",
        "work_hours",
        "late_minutes",
        "overtime_minutes",
        "half_day",
        "history",
    )
    query = {"_id": document["_id"]}
    for field_name in fields:
        if field_name in document:
            query[field_name] = document[field_name]
        else:
            query[field_name] = {"$exists": False}
    return query


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, datetime) and isinstance(right, datetime):
        return as_utc(left) == as_utc(right)
    return left == right


def _history_value(field_name: str, value: object) -> object:
    if field_name in ("punch_in", "punch_out"):
        return datetime_to_epoch_ms(value) if value is not None else None
    return value


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str, date: date, body: RegularizeRequest
) -> dict:
    day = date.isoformat()
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")
    original = db.attendance_logs.find_one({"emp_code": emp_code, "date": day})
    if original is None:
        raise HTTPException(404, "attendance record not found")

    final_status: Status = body.status or original["status"]
    original_punch_in = original.get("punch_in")
    original_punch_out = original.get("punch_out")
    normalized_original_punch_in = (
        truncate_to_second(original_punch_in) if original_punch_in is not None else None
    )
    normalized_original_punch_out = (
        truncate_to_second(original_punch_out) if original_punch_out is not None else None
    )
    supplied_punch_in = (
        epoch_ms_to_datetime(body.punch_in)
        if body.punch_in is not None
        else normalized_original_punch_in
    )
    supplied_punch_out = (
        epoch_ms_to_datetime(body.punch_out)
        if body.punch_out is not None
        else normalized_original_punch_out
    )

    if final_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                422, "ABSENT and LEAVE records cannot include punch times"
            )
        final_punch_in = None
        final_punch_out = None
    else:
        final_punch_in = supplied_punch_in
        final_punch_out = supplied_punch_out
        if final_punch_in is None:
            raise HTTPException(422, "presence status requires punch_in")
        if attendance_date(final_punch_in, employee).isoformat() != day:
            raise HTTPException(
                422, "punch_in must belong to the record's attendance date"
            )
        if final_punch_out is not None:
            duration = final_punch_out - final_punch_in
            if duration.total_seconds() <= 0 or duration > timedelta(hours=24):
                raise HTTPException(
                    422, "punch_out must be after punch_in and within 24 hours"
                )

    derived = derive_attendance_values(
        employee, final_status, day, final_punch_in, final_punch_out
    )
    final_values = {
        "status": final_status,
        "punch_in": final_punch_in,
        "punch_out": final_punch_out,
        **derived,
    }
    old_values = {
        "status": original.get("status"),
        "punch_in": original.get("punch_in"),
        "punch_out": original.get("punch_out"),
        "work_hours": original.get("work_hours"),
        "late_minutes": original.get("late_minutes", 0),
        "overtime_minutes": original.get("overtime_minutes", 0),
        "half_day": original.get("half_day", False),
    }
    changes = {}
    for field_name, new_value in final_values.items():
        old_value = old_values[field_name]
        if not _same_value(old_value, new_value):
            changes[field_name] = {
                "from": _history_value(field_name, old_value),
                "to": _history_value(field_name, new_value),
            }
    if not changes:
        raise HTTPException(422, "regularization does not change the record")

    history_entry = {
        "at": now_utc_millis(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    result = db.attendance_logs.update_one(
        _snapshot_filter(original),
        {
            "$set": final_values,
            "$push": {"history": history_entry},
        },
    )
    if result.modified_count != 1:
        raise HTTPException(409, "record changed during regularization; retry")
    updated = {
        **original,
        **final_values,
        "history": [*(original.get("history") or []), history_entry],
    }
    return attendance_response(updated)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: Annotated[str, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")],
) -> dict:
    pipeline = build_employee_monthly_pipeline(emp_code, month)
    result = next(db.employees.aggregate(pipeline, allowDiskUse=True), None)
    if result is None:
        raise HTTPException(404, "employee not found")
    return result


@app.get("/analytics/departments/summary")
def department_summary(
    month: Annotated[str, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")],
    department: str | None = None,
) -> dict:
    pipeline = build_department_summary_pipeline(month, department)
    items = list(db.employees.aggregate(pipeline, allowDiskUse=True))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: Annotated[str, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    department: str | None = None,
) -> dict:
    pipeline = build_late_leaderboard_pipeline(month, limit, department)
    items = list(db.attendance_logs.aggregate(pipeline, allowDiskUse=True))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_day: date = Query(alias="from"),
    to_day: date = Query(alias="to"),
) -> dict:
    if to_day < from_day:
        raise HTTPException(422, "to must not be before from")
    if (to_day - from_day).days + 1 > 92:
        raise HTTPException(422, "trend range cannot exceed 92 days")
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "department not found")
    pipeline = build_department_trend_pipeline(department, from_day, to_day)
    items = list(db.employees.aggregate(pipeline, allowDiskUse=True))
    return {"department": department, "items": items}


def _explain(collection_name: str, pipeline: list[dict]) -> dict:
    return db.command(
        "explain",
        {
            "aggregate": collection_name,
            "pipeline": pipeline,
            "cursor": {},
            "allowDiskUse": True,
        },
        verbosity="executionStats",
    )


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal[
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ],
    emp_code: str | None = None,
    month: Annotated[
        str | None, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    ] = None,
    department: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    date_from: date | None = None,
    date_to: date | None = None,
    status: Status | None = None,
    from_day: date | None = Query(default=None, alias="from"),
    to_day: date | None = Query(default=None, alias="to"),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    if endpoint == "attendance_list":
        if date_from is not None and date_to is not None and date_from > date_to:
            raise HTTPException(422, "date_from must not be after date_to")
        query = build_attendance_query(emp_code, date_from, date_to, status)
        explanation = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": query,
                "projection": {"_id": 0},
                "sort": {"date": DESCENDING, "emp_code": ASCENDING},
                "skip": (page - 1) * page_size,
                "limit": page_size,
            },
            verbosity="executionStats",
        )
        collection_name = "attendance_logs"
    elif endpoint == "employee_monthly":
        if not emp_code or month is None:
            raise HTTPException(422, "emp_code and month are required")
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        explanation = _explain("employees", pipeline)
        collection_name = "employees"
    elif endpoint == "department_summary":
        if month is None:
            raise HTTPException(422, "month is required")
        pipeline = build_department_summary_pipeline(month, department)
        explanation = _explain("employees", pipeline)
        collection_name = "employees"
    elif endpoint == "late_leaderboard":
        if month is None:
            raise HTTPException(422, "month is required")
        pipeline = build_late_leaderboard_pipeline(month, limit, department)
        explanation = _explain("attendance_logs", pipeline)
        collection_name = "attendance_logs"
    else:
        if department is None or from_day is None or to_day is None:
            raise HTTPException(422, "department, from, and to are required")
        if to_day < from_day:
            raise HTTPException(422, "to must not be before from")
        if (to_day - from_day).days + 1 > 92:
            raise HTTPException(422, "trend range cannot exceed 92 days")
        pipeline = build_department_trend_pipeline(department, from_day, to_day)
        explanation = _explain("employees", pipeline)
        collection_name = "employees"

    json_explanation = json.loads(json_util.dumps(explanation))
    return {
        "endpoint": endpoint,
        "collection": collection_name,
        "explain": json_explanation,
    }
