# PesaShield message checker

Paste one complete M-PESA message at `/demo#pesashield` or in the signed-in live workspace. The form calls `POST /api/pesashield/sms/check`; it accepts `text`, optional `sender`, and optional `expected_amount` in KES. The analysis runs locally on the server and does not call a hosted language-model API.

The checker reads the transaction reference, amount, transaction type, counterparty, date/time and stated balance. Masked phone numbers and normal Paybill receipts are supported. An optional expected amount can reveal a mismatch. Missing or invalid information is retained as uncertainty instead of being treated as a genuine payment.

Four outcomes are possible:

- **High fraud risk:** strong text indicators, such as a request to disclose a PIN/OTP, send a direct reversal payment, or pay a prize/loan release fee.
- **Suspicious - verify first:** a link, changed payment destination, invalid date, multiple receipts, amount mismatch or another concern needs independent checking.
- **Looks legitimate:** a recognised receipt structure with no listed red flags. This is not an authenticated transaction.
- **Unable to determine:** insufficient receipt information and no clear fraud evidence. Unrelated text does not receive a legitimate verdict.

Negated safety reminders such as “Never share your PIN” are distinguished from disclosure requests. Rules are limited pattern checks, not comprehensive fraud detection. The existing TF-IDF/logistic regression model can add supporting caution to an already suspicious message; its tiny starter dataset is not sufficient to override a receipt or establish fraud probability. The UI shows evidence rather than an uncalibrated percentage.

Every result says **Payment not verified**. The app is not connected to a Safaricom transaction-verification service and cannot tell a perfectly copied receipt from an authentic one. Compare the message with your own transaction records before acting.

Public demo checks and checks in unprotected workspaces are stateless: message bodies are not saved. Only a protected, signed-in live workspace saves private history. Identical retries count once, including requests without a sender. Team labels are separate from the original assessment and do not authenticate payment. Old assessments are marked for rechecking.

Sources: [Safaricom Buy Goods security and verification guidance](https://www.safaricom.co.ke/images/Downloads/Buy-goods-guides.pdf), [Safaricom transaction-status API](https://developer.safaricom.co.ke/apis/TransactionStatus). The latter is a reference for future authenticated verification, not a connection implemented by this update.
