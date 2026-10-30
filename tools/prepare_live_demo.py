"""Create private local configuration or report readiness without printing secrets."""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values, load_dotenv, set_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Report configuration readiness only; no provider network requests.')
    args = parser.parse_args()
    path = ROOT / '.env'
    if not args.check:
        if not path.exists(): path.write_text((ROOT / '.env.example').read_text())
        path.chmod(0o600)
        values = dotenv_values(path)
        for key in ('GHOST_SECRET_KEY', 'WHATSAPP_VERIFY_TOKEN'):
            if not values.get(key) and not os.environ.get(key):
                set_key(path, key, secrets.token_urlsafe(48))
        print(f'Private configuration prepared at {path}. No credential values are printed.')
        print('Fill the workspace password, Meta app secret/phone ID, Gmail account/app password and public HTTPS URL privately.')
        return
    load_dotenv(path, override=False)
    from live_channels import gmail_config, public_base
    from workspace_auth import privacy_ready
    fields = ('GHOST_SECRET_KEY', 'GHOST_ADMIN_PASSWORD', 'WHATSAPP_APP_SECRET',
              'WHATSAPP_PHONE_NUMBER_ID', 'WHATSAPP_VERIFY_TOKEN', 'GHOST_GMAIL_ACCOUNT',
              'GHOST_GMAIL_APP_PASSWORD', 'GHOST_PUBLIC_URL')
    print(json.dumps(dict(private_workspace_ready=privacy_ready(), gmail_configuration_ready=gmail_config() is not None,
                          public_https_url_ready=bool(public_base()),
                          meta_configuration_present=all(os.environ.get(key) for key in fields[2:5]),
                          missing_fields=[key for key in fields if not os.environ.get(key)]), indent=2))


if __name__ == '__main__': main()
