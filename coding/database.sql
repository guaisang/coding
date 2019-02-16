-- top n in each group
select new_bill, rate_plan_type, rankBill from
(select new_bill, rate_plan_type, row_number() over (partition by rate_plan_type order by new_bill desc) as rankBill from user_dw.VS_SC_Campaign_Monthly) a
where rankBill < 10

-- top 3 in each group without rank(1124)/row_number(1234)/dense_rank(1123)
SELECT
    d.Name AS 'Department', e1.Name AS 'Employee', e1.Salary
FROM
    Employee e1
        JOIN
    Department d ON e1.DepartmentId = d.Id
WHERE
    3 > (SELECT
            COUNT(DISTINCT e2.Salary)
        FROM
            Employee e2
        WHERE
            e2.Salary > e1.Salary
                AND e1.DepartmentId = e2.DepartmentId
        )
;


-- calculate median in each group
select a.new_bill, rate_plan_type, billRank, totalRows from
(select new_bill, rate_plan_type, count(1) over (partition by rate_plan_type) as totalRows,
row_number() over (partition by rate_plan_type order by new_bill asc) as billRank from user_dw.VS_SC_Campaign_Monthly) a
where a.billRank = round(a.totalRows/2.0,0);

-- Declare variable
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE M INT;
SET M=N-1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT DISTINCT Salary FROM Employee ORDER BY Salary DESC LIMIT M, 1
  );
END

-- Find all numbers that appear at least three times concecutively
select distinct a.Num as ConsecutiveNums
from logs a inner join logs b on a.Id=b.Id+1 and a.Num = b.Num
inner join logs c on b.Id = c.Id+1 and b.Num = c.Num;

-- 183. Department Highest Salary
select c.Name as Department, b.Name as Employee, b.Salary from Employee b inner join
(select max(Salary) as Salary, DepartmentId from Employee
group by DepartmentId) a
on b.DepartmentId = a.DepartmentId and b.Salary = a.Salary
inner join Department c on a.DepartmentId=c.Id;

-- switch id with neighbor
select
case when s.id is null then a.id-1 else a.id end as id, a.student
from
(select id-1 as id, student from seat
where id%2=0
union
select id+1 as id, student from seat
where id%2=1
order by id asc) a
left join seat s on a.id=s.id;

-- calculate rate
select Request_at as Day,
round(sum(case when Status = 'completed' then 0 else 1 end)/count(*), 2) as 'Cancellation Rate'
from Trips
where Client_id in (select Users_Id from Users where Banned='No' and Role='client') and
Driver_Id in (select Users_Id from Users where Banned='No' and Role='driver')
and Request_at between '2013-10-01' and '2013-10-03'
group by Request_at;

-- consecutive day with number greater than 100
select distinct t1.*
from stadium t1, stadium t2, stadium t3
where t1.people >= 100 and t2.people >= 100 and t3.people >= 100
and
(
	  (t1.id - t2.id = 1 and t1.id - t3.id = 2 and t2.id - t3.id =1)  -- t1, t2, t3
    or
    (t2.id - t1.id = 1 and t2.id - t3.id = 2 and t1.id - t3.id =1) -- t2, t1, t3
    or
    (t3.id - t2.id = 1 and t2.id - t1.id =1 and t3.id - t1.id = 2) -- t3, t2, t1
)
order by t1.id
;

-- calculate the difference between current value and previous value
SELECT user_id, unix_timestamp - previous_time AS Delta_SecondLast0ne_LastOne
FROM
(SELECT user_id, unix_timestamp,
LAG(unix_timestamp, 1) OVER (PARTITION BY user_id ORDER BY unix_timestamp) AS previous_time,
ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY unix_timestamp DESC) AS order_desc
FROM query_one
) tmp
WHERE order_desc = 1
ORDER BY user_id

-- calculate percentage
SELECT 100*SUM(CASE WHEN m.user_id IS null THEN 1 ELSE 0 END)/COUNT(*) as
WEB_ONLY,
100*SUM(CASE WHEN w.user_id IS null THEN 1 ELSE 0 END)/COUNT(*) as
MOBILE_ONLY,
100*SUM(CASE WHEN m.user_id IS NOT null AND w.user_id IS NOT null
THEN 1 ELSE 0 END)/COUNT(*) as BOTH
FROM
(SELECT distinct user_id FROM query_two_web ) w
FULL OUTER JOIN
(SELECT distinct user_id FROM query_two_mobile ) m
ON m.user_id = w.user_id;

-- cumulative value up to each date
SELECT user_id, date,
SUM(amount) over(PARTITION BY user_id ORDER BY date) as total_amount
FROM
(
SELECT user_id, date, SUM(transaction_amount) as amount
FROM query_four_march
GROUP BY user_id, date
UNION ALL
SELECT user_id, date, SUM(transaction_amount) as amount
FROM query_four_april
GROUP BY user_id, date
) tmp
ORDER BY user_id, date

-- delete rows
delete from Person where Id not in
(select Id from(select min(Id) as Id from Person
 group by Email) a)

-- update values in a column
update salary
set sex=
case when sex='f' then 'm'
else 'f' end
