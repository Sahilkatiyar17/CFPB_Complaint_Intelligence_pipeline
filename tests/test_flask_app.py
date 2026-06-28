import unittest
import os
import sys

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))

from flask_app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(
            text="I was charged twice on my credit card and the bank refused to refund me."
        ))
        self.assertEqual(response.status_code, 200)

        # Match your actual app output
        self.assertTrue(
            b'Credit Card Services' in response.data or
            b'Bank Accounts and Services' in response.data or
            b'Credit Reporting' in response.data or
            b'Debt Collection' in response.data or
            b'Loans' in response.data,
            "Response should contain a product category"
        )

    def test_urgency_in_response(self):
        response = self.client.post('/predict', data=dict(
            text="My account has been frozen and I cannot pay my rent."
        ))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Yes' in response.data or b'No' in response.data,
            "Response should contain urgency prediction"
        )

if __name__ == '__main__':
    unittest.main()