# Betfair SSL Certificates

Place your Betfair Exchange API SSL certificates here:

- `client-2048.crt`
- `client-2048.key`

## How to obtain

1. Log into [betfair.com](https://www.betfair.com)
2. Go to: **Account → My Account → API Developer Programme**
3. Create a **Delayed Data** application key (free — 3-minute delay on odds, which is fine for pre-match value bets)
4. Download the SSL certificate files and place them in this directory

## Environment variables

Set these in `.env` (copy from `.env.example`):

```
BETFAIR_USERNAME=samhusbands@live.co.uk
BETFAIR_PASSWORD=YoggSaron75!
BETFAIR_APP_KEY=fpqkEdjYtHDc9uxU
BETFAIR_CERT_PATH=certs/client-2048.crt
BETFAIR_KEY_PATH=certs/client-2048.key
```

These are also set as **GitHub Actions Secrets** for the daily pipeline:
- `BETFAIR_USERNAME`
- `BETFAIR_PASSWORD`
- `BETFAIR_APP_KEY`
- `BETFAIR_CERT` (full contents of .crt file)
- `BETFAIR_KEY` (full contents of .key file)

The GitHub Actions workflow writes the cert contents to files at runtime.

## Security

The `.gitignore` excludes `*.crt`, `*.key`, and `*.pem` files so credentials
are never committed to the repository.
