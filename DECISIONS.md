# DECISIONS.md

1. Indexes. A unique employees.emp_code index protects employee identity. A unique attendance_logs(emp_code, date) index guarantees one daily record and supports per-employee lookups. Employee department/joined_on and joined_on indexes support headcount; department/emp_code supports sorted department lists. Attendance date/emp_code supports pages, status/date/emp_code supports filtered pages, emp_code/punch_in/date locates the latest punch, and date/late_minutes/emp_code supports the leaderboard. I rejected history.at because no endpoint searches audit entries.

2. Punch-in race. Both requests look up the employee and attempt an insert. The unique natural-key index accepts one; MongoDB rejects the other with DuplicateKeyError, which becomes 409. Exactly one request returns 201.

3. Ties. The pipeline ranks by total late minutes, filters rank <= limit, then orders tied rows by emp_code. Everyone tied at an eligible rank is returned, so the result can contain more than limit rows.

4. Headcount. The summary starts from employees who joined by month end, then left-looks up monthly logs. Preserving an empty lookup row counts an employee with no logs once.

5. At 100x, I would maintain daily department rollups when records change, while retaining the source logs and audit history. That would reduce repeated cross-collection aggregation work.
