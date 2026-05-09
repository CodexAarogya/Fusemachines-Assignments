from locust import HttpUser, task, constant


class CustomerUser(HttpUser):

    wait_time = constant(0)

    @task
    def get_customer(self):
        self.client.get(
            "/api/v1/customers/customer?customerNumber=103"
        ) 