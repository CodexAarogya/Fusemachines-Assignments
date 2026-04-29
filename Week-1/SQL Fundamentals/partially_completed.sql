use classicmodels;
-- Show all the customers whose creditLimit is greater than 20000 
select customerName from customers
where creditLimit > 20000;

-- Show the employees who report to VP Sales.
select firstName from employees
where reportsTo  = (select employeeNumber from employees where jobTitle = 'VP Sales');

-- Find all the customers who have set their state while filling the forms and Lives in USA and credit limit is between 100000 and 200000.
select customerName, state, creditLimit from customers
where state is not null 
and country = 'USA'
and creditLimit between 100000 and 200000;

-- Find all the employees who report to Sales Managers of all types.
select firstName, lastName, reportsTo from employees
where reportsTo in (select employeeNumber from employees where jobTitle like '%Sale%Manager%');

-- Find the average credit limit of customers of each country.
select country, avg(creditLimit) as Avg_credit_Limit from customers
group by country;

-- Find the total no. of orders for each date and customer. Show only dates with total number of orders greater than 10 for date and customer.
select t1.orderDate from
(select orderDate, count(orderNumber) as total_orders, count(customerNumber) as total_customers from orders
group by orderDate having total_orders > 10) as t1;

-- Find the name of the supervisor, job title of supervisor and total no. of supervisee using subquery. (With out using Join operation)
select * from employees;

-- Find all customers with a credit limit greater than average credit credit limit using WITH Clause.
with credit as
(select avg(creditLimit) as avg_credit_limit from customers)
select customerName from customers, credit
where customers.creditLimit > credit.avg_credit_limit;

-- Find the rank of customer. [Customer with highest credit limit have 1 rank and Customer with lowest credit limit have highest rank]. Then, find the customer with the third highest credit limit.
with credit_ranks as 
(select customerName, creditLimit, rank() over (order by creditLimit desc) as rank_no from customers)
select customerName, creditLimit from credit_ranks
where rank_no = 3;

-- Create a view showing the customer name, complete address, and their total number of orders.
create view customer_order_details as
(with customer_orders as
(select customerNumber, count(orderNumber) as total_orders from orders 
group by customerNumber)
select customers.customerName, customers.addressLine1, customers.addressLine2, customer_orders.total_orders from customers
inner join customer_orders
on customers.customerNumber = customer_orders.customerNumber);

-- Update the country of a customer (use any one record).
update customers
set country = 'Nepal'
where customerNumber = 103;
select * from customers;