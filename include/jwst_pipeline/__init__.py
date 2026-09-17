"""
JWST golden dataset pipeline.

Everything that does real work lives in this package, and none of it imports Airflow,
so all of it can be tested without a scheduler. The DAGs in dags/ decide ordering,
retries and concurrency, and call in here for the rest.
"""
