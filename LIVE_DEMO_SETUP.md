# Live WhatsApp + Gmail presentation

The integration code is installed in GhostDemand. Account connection is complete only after your private configuration is set and real test messages arrive. The app receives messages; it does not send WhatsApp replies or emails.

## 1. Private server configuration

From the project directory, run:

```sh
.venv/bin/python tools/prepare_live_demo.py
```

This creates a Git-ignored `.env` file with owner-only file permissions, preserving any existing file and values. A random signing key and webhook verification token are generated locally. Open `.env` in your editor and fill the following privately:

| Variable | Value |
| --- | --- |
| `GHOST_ADMIN_PASSWORD` | A new workspace password of at least 12 characters. This is separate from your Google password. |
| `WHATSAPP_APP_SECRET` | The app secret for the Meta app that owns your WhatsApp Cloud API integration. |
| `WHATSAPP_PHONE_NUMBER_ID` | The selected WhatsApp Business phone-number ID, rather than its display phone number. |
| `GHOST_GMAIL_ACCOUNT` | The Gmail account receiving your presentation enquiries. |
| `GHOST_GMAIL_APP_PASSWORD` | A Google app password, without needing the normal Google account password. |
| `GHOST_PUBLIC_URL` | Your deployed app's public HTTPS root URL, such as the actual URL shown by Render. |

Keep the generated `GHOST_SECRET_KEY` and `WHATSAPP_VERIFY_TOKEN`. Do not paste any secrets into chat or commit `.env`. On Render, enter these fields in the service's private **Environment** settings; a local `.env` is not uploaded. Keep live and demo databases on separate paths.

Check readiness without contacting providers or printing credential values:

```sh
.venv/bin/python tools/prepare_live_demo.py --check
```

Restart the app after changing configuration. In particular, do not import real mail into a live database while an older, unprotected server process is still serving it.

## 2. Gmail: dedicated demo label

1. Use a dedicated presentation mailbox, or create the Gmail label **`GhostDemandDemo`** in the account you selected. Keep `GHOST_GMAIL_FOLDER=GhostDemandDemo`. The listener deliberately refuses `INBOX` and All Mail.
2. In Gmail settings, create a filter for a distinctive subject such as **`[GhostDemand Demo]`**, and apply the `GhostDemandDemo` label to matching incoming messages. Do not apply it to all existing conversations.
3. Ensure the label is visible to IMAP clients in Gmail's label settings, where that control is available. Personal Gmail IMAP access is already enabled; you do not need to find an old “Enable IMAP” switch. [Google's current Gmail client guidance](https://support.google.com/mail/answer/7126229?hl=en).
4. With 2-Step Verification enabled, create an app password in your Google account and place it only in the server's private configuration. Google recommends OAuth for longer-term applications; this single-account demo uses the app-password option you confirmed is available. [Google app-password instructions and restrictions](https://support.google.com/accounts/answer/185833?hl=en).
5. Start the updated server and sign in at `/login` with the workspace password. Open **Channels & import**. Wait until Gmail shows **Listening**, or click **Check new email**.

On the first successful connection, the listener records the label's current UID position and imports **no existing messages**. Send your test enquiry only after it says Listening. It checks approximately every 15 seconds; the dashboard refreshes every eight seconds. Allow around 25 seconds for a stage demonstration under normal network conditions. This timing is a configured polling interval, not a delivery guarantee.

The listener uses TLS, selects the label read-only, and fetches message bodies with `BODY.PEEK`. It does not mark messages read, modify labels, delete email or download attachments. Large, truncated or attachment-only messages wait for context. A failed analysis is retained and retried without advancing past that message. A database lease prevents multiple workers from importing the same batch concurrently.

Send from another account or ask a teammate to send:

```text
Subject: [GhostDemand Demo] Wireless earbuds

I need 2 wireless earbuds, but your website says they are sold out.
My total budget is KES 6,000 and I need them today.
```

## 3. WhatsApp: Meta Cloud API webhook

This adapter targets **Meta's Cloud API**, not the regular WhatsApp Business phone app or a provider-specific alternative. If your existing API is hosted by a business solution provider, confirm its webhook format before replacing any provider configuration.

1. Deploy this source to your existing HTTPS web service and set the private variables above. Open the live workspace's **Channels & import** page to see the exact callback URL:

   ```text
   https://YOUR-ACTUAL-APP-HOST/webhooks/whatsapp
   ```

2. In the Meta app attached to your WhatsApp Business account, configure the WhatsApp webhook callback with that URL and the same private `WHATSAPP_VERIFY_TOKEN` value. Subscribe to incoming **messages** events for the selected business phone number. Keep the current callback configuration available so you can restore it if the existing integration must continue elsewhere.
3. Meta's verification request must receive the challenge as plain text. The app then verifies each delivery's `X-Hub-Signature-256` HMAC against the **app secret**, and accepts only the selected phone-number ID. A verification token and an app secret have different roles. [Meta's official signature-validation sample](https://github.com/fbsamples/whatsapp-api-examples/tree/main/signature-validation-with-webhooks-payloads).
4. From an allowed test phone or a real sender permitted by your current Meta account mode, send this text to your WhatsApp Business number:

   ```text
   I need 2 wireless earbuds, but they are sold out. Budget KES 6,000. I need them today.
   ```

5. Open `/dashboard#activity`, not the synthetic `/demo` workspace. The event should show **WhatsApp**, its original text, and its captured demand or Needs context outcome. Re-delivery of the same provider message ID counts once.

Delivery/read-status notifications, messages for a different number, and media-only messages do not create demand signals. This first integration supports incoming **text** enquiries. It does not download audio, transcribe voice messages or send automatic replies.

## 4. Stage sequence

1. Open the private live workspace and sign in before the presentation. Keep the app visible so automatic refreshes continue. Confirm Gmail says Listening and WhatsApp is verified.
2. Send the labelled email. While it arrives, explain the problem in one sentence.
3. Send the WhatsApp message from your phone. Show both source labels in Channel activity.
4. Open each request to show the extracted product, quantity, budget and blocker. The model and its training details stay on the backend.
5. Open Opportunities to show the related product demand, then assign an owner and next action.

Use your own harmless test enquiries. New real messages enter the live database and are visible only in the signed-in workspace. The public demo uses a separate database and contains synthetic examples. Contact references are minimized; the source platforms remain the place to contact the sender. Ordinary availability questions without an established missed outcome stay in Needs context.

## 5. Readiness and limits

- `Webhook verified` proves Meta completed its callback check. `Messages received` requires an actual validated incoming text delivery.
- `Listening` means Gmail successfully selected the demo label. Complete the test with a newly delivered, labelled message before claiming end-to-end success.
- The automatic intake path uses local backend extraction and classification, including when an optional hosted language-model key exists elsewhere in the app. Private channel messages are not forwarded to a hosted language-model API by these connectors.
- The live workspace requires a password and stable signing key before either connector accepts private messages. Browser changes use CSRF protection; provider callbacks use their own signature validation.
- Credentials stay in environment configuration, not frontend JavaScript, chat, Git or database state. Gmail cursors and safe connection status are stored in SQLite. No provider credentials are returned by the status API.
- The built-in login is for a single workspace. Longer-term use needs the business's identity, access and operating policies, and representative model evaluation. Use durable storage on the hosted service if history and mailbox cursors must survive runtime replacement. No paid storage or provider plan is purchased by this setup.
