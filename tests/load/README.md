# Load Testing

This directory contains the first Locust scaffold for Montferrand.

Recommended setup for repeatable runs:

1. Run the app with a deterministic fake `process_message` and fake `_send_sms`.
2. Provision at least one tenant before starting Locust.
3. Point Locust at the running server and set:
   - `MONTFERRAND_LOAD_TENANT_PHONE`
   - `MONTFERRAND_LOAD_CUSTOMER_PREFIX`

Example:

```sh
locust -f tests/load/locustfile.py --host http://127.0.0.1:8080
```

The current script focuses on `/sms` ingestion pressure. The next iteration
should add a dedicated harness that stubs the model and Twilio sender while
asserting CRM and calendar side effects under load.
