"""Generate a synthetic, anonymized dataset of vendor replies to GST mismatch emails.

Why synthetic: the real replies this project is modelled on belong to a client and
contain names, GSTINs and invoice numbers, so they cannot be published. The intent
categories and the way vendors phrase things are modelled on that real workflow;
every name, number and sentence here is made up.

Why templates are split into train and test groups: if the same template appeared
on both sides, a model could score near 100% by memorizing phrasing. Each class
has its own held-out templates, so the test set measures generalization to
wording the model never saw.

Run:  python data/generate_dataset.py   ->  data/vendor_replies.csv
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
SAMPLES_PER_TEMPLATE = 12
TEST_TEMPLATES_PER_CLASS = 3  # the last N templates of each class are test-only

# Each class: list of templates. Slots: {doc} {month} {amt} {gstin} {days}
TEMPLATES: dict[str, list[str]] = {
    "WILL_AMEND": [
        "We have noted the difference in invoice {doc} and will amend it in the {month} return.",
        "This was missed by our accounts team by mistake. We shall rectify it while filing GSTR-1 for {month}.",
        "Kindly give us some time, the correction for {doc} will reflect in next month's filing.",
        "We will upload the missing invoice {doc} in our upcoming return for {month}.",
        "Our consultant will correct the tax amount for {doc} in the coming month.",
        "Noted. Amendment for {doc} will be done in the subsequent return period.",
        "We accept the mismatch and will fix it in the next GSTR-1, which is due in {month}.",
        "The invoice {doc} was not reported earlier; it will be included when we file for {month}.",
        "Please allow us till {month} to amend the details of {doc} on the portal.",
        "We are aware of the gap and plan to report {doc} in our {month} filing without fail.",
    ],
    "ALREADY_CORRECTED": [
        "The invoice {doc} has already been amended in our GSTR-1 for {month}. Please check 2B again.",
        "We have corrected the GSTIN on {doc} and filed the amendment last week.",
        "This issue is already resolved. The revised invoice {doc} is now visible on the portal.",
        "Amendment done for {doc} in the {month} return. Kindly verify at your end.",
        "We filed the correction for {doc} on time, it should reflect in your 2A now.",
        "The tax difference of Rs {amt} on {doc} was fixed in our last filing.",
        "Please recheck, {doc} was rectified in the {month} return and the status shows filed.",
        "We already updated the taxable value for {doc} on the GST portal.",
        "Correction for invoice {doc} has been completed and is reflecting in GSTR-2B.",
        "The mismatch was sorted out in {month}; nothing is pending from our side for {doc}.",
    ],
    "DOCUMENTS_SUBMITTED": [
        "Please find attached the invoice copy for {doc} as requested.",
        "Attached are the tax invoice and e-way bill for {doc}. Kindly check.",
        "As discussed, we are sharing the signed copy of invoice {doc} with this mail.",
        "PFA the scanned invoice {doc} along with the delivery challan.",
        "Enclosed herewith the documents for {doc} for your verification.",
        "We have attached the GSTR-1 filing screenshot showing {doc}.",
        "Sharing the invoice PDF for {doc}; please confirm receipt.",
        "Kindly see the attachment, it contains the invoice and credit note for {doc}.",
        "Here is the copy of {doc} you asked for, attached in PDF format.",
        "Attaching the relevant invoice {doc} and our ledger extract for {month}.",
    ],
    "DISPUTES_DATA": [
        "We do not agree with the difference shown. As per our books, {doc} is correctly reported.",
        "Your records seem incorrect. Invoice {doc} was billed at Rs {amt}, not the amount you mention.",
        "There is no such invoice {doc} raised by us. Please check your purchase register.",
        "The tax amount in your mail does not match our ledger. Kindly recheck {doc}.",
        "We have reported {doc} correctly in GSTR-1; the error is in your booking.",
        "This mismatch is due to wrong entry at your end, our filing for {doc} is accurate.",
        "We dispute the figures shared. Our ledger shows Rs {amt} against {doc}.",
        "Please verify your data, we never issued invoice {doc} in {month}.",
        "The amount of Rs {amt} you have booked is wrong; our invoice {doc} says otherwise.",
        "According to our reconciliation, nothing is wrong with {doc}. Please review your entries.",
    ],
    "CREDIT_NOTE_EXPLANATION": [
        "The difference in {doc} is because we issued a credit note of Rs {amt} against it.",
        "A credit note was raised for rate difference on {doc}, which explains the mismatch.",
        "The gap of Rs {amt} is due to the credit note issued in {month} against {doc}.",
        "We had issued a debit note for {doc}; that is why the values differ.",
        "Credit note against {doc} was reported separately in GSTR-1, hence the lower value.",
        "The reduction on {doc} is on account of a discount credit note of Rs {amt}.",
        "Part quantity of {doc} was returned and a credit note was issued for the same.",
        "Please adjust the credit note raised against {doc}; it covers the full difference.",
        "The mismatch reflects the credit note for short supply against invoice {doc}.",
        "We passed a credit note in {month} for {doc}, so net tax is lower than your books.",
    ],
    "STATUS_REQUEST": [
        "Can you please confirm whether the issue for {doc} is resolved now?",
        "We are awaiting your confirmation on the status of {doc}. Please update.",
        "Kindly let us know if anything is still pending from our side.",
        "Please share the current status of the reconciliation for {month}.",
        "Is there any further action required from us for invoice {doc}?",
        "Request you to confirm if our earlier correction for {doc} is accepted.",
        "We have not heard back. Is the mismatch for {doc} closed?",
        "Kindly confirm whether you need anything more for {doc}.",
        "Could you update us on whether the {month} differences are cleared?",
        "Please advise if the matter regarding {doc} has been closed at your end.",
    ],
    "WRONG_BUYER_GSTIN": [
        "Invoice {doc} was raised to a different GSTIN by mistake. We will issue a corrected one.",
        "The invoice {doc} belongs to your other plant; it was billed to GSTIN {gstin}.",
        "This invoice was not for your company, it was raised to another customer.",
        "We wrongly mentioned buyer GSTIN {gstin} on {doc}. Kindly ignore it.",
        "Invoice {doc} was billed to your unit in another state, please check that GSTIN.",
        "The buyer name on {doc} is incorrect, it should have been your company.",
        "We billed {doc} under GSTIN {gstin} in error and are correcting the party details.",
        "This document is for a different branch of yours, registered under {gstin}.",
        "Please note {doc} was issued to another party, not to your organisation.",
        "The ship-to and bill-to GSTIN on invoice {doc} were swapped by mistake.",
    ],
    "CONTACT_CHANGE": [
        "Please note our accounts contact has changed. Kindly write to the new team going forward.",
        "I have moved to another role. Please follow up with our finance team for {doc}.",
        "Kindly update our email ID in your records; the old mailbox is no longer monitored.",
        "Adding our GST team in this mail, they will handle the reconciliation from now.",
        "Please remove me from this thread and send future reminders to our accounts head.",
        "Our company has a new point of contact for GST matters from {month}.",
        "The person handling this has left; please mark our tax consultant in future mails.",
        "Please update the vendor contact details, the new email is copied here.",
        "Kindly direct all GST queries to our central accounts mailbox from now on.",
        "I am no longer handling this vendor account; my colleague will reply on {doc}.",
    ],
    "ACKNOWLEDGEMENT": [
        "Thank you for the information. We have noted it.",
        "Received your mail, we will check and revert.",
        "Noted with thanks.",
        "Thanks for the update, will look into it.",
        "We acknowledge receipt of your reminder for {month}.",
        "Okay, noted. Our team will review.",
        "Thank you, we have forwarded this to the concerned person.",
        "Received with thanks. Will get back shortly.",
        "Noted your mail regarding {doc}, we will check.",
        "Acknowledged. We will respond after internal review.",
    ],
    "MEETING_REQUEST": [
        "Can we schedule a call to discuss the differences for {month}?",
        "It will be easier to resolve this over a meeting. Please share a convenient time.",
        "Please arrange a call with your GST team to discuss {doc}.",
        "We would like to meet your accounts team to reconcile all open invoices.",
        "Kindly share your phone number so we can discuss {doc} directly.",
        "Request a short video call this week to close the pending mismatches.",
        "Can someone from your side join a call on the {month} reconciliation?",
        "Let us discuss over a call, the mail thread for {doc} is getting long.",
        "We propose a meeting within {days} days to settle the open differences.",
        "Please let us know when your team is available to connect on {doc}.",
    ],
}

MONTHS = ["April", "May", "June", "July", "August", "September", "October", "November",
          "December", "January", "February", "March"]
OPENERS = ["", "Dear Sir, ", "Dear Team, ", "Hello, ", "Dear Madam, ", "Hi, ", "Respected Sir, "]
CLOSERS = ["", " Regards.", " Thanks and regards.", " Thank you.", " Best regards, Accounts Team.", " Regards, Finance Dept."]


def _doc(rng: random.Random) -> str:
    prefix = rng.choice(["INV", "TI", "GST", "SI", "BL"])
    sep = rng.choice(["-", "/", ""])
    return f"{prefix}{sep}{rng.randint(1000, 99999)}"


def _gstin(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    state = f"{rng.randint(1, 37):02d}"
    pan = "".join(rng.choice(letters) for _ in range(5)) + f"{rng.randint(1000, 9999)}" + rng.choice(letters)
    return f"{state}{pan}1Z{rng.choice(letters)}"


def _maybe_noise(text: str, rng: random.Random) -> str:
    """Light, realistic noise: lowercase mail, a dropped word, or a typo."""
    r = rng.random()
    if r < 0.15:
        return text.lower()
    if r < 0.25:
        words = text.split()
        if len(words) > 6:
            del words[rng.randrange(1, len(words) - 1)]
        return " ".join(words)
    if r < 0.33:
        i = rng.randrange(1, max(2, len(text) - 1))
        return text[:i] + text[i + 1:]
    return text


def generate(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for label, templates in TEMPLATES.items():
        split_at = len(templates) - TEST_TEMPLATES_PER_CLASS
        for t_idx, template in enumerate(templates):
            split = "test" if t_idx >= split_at else "train"
            for _ in range(SAMPLES_PER_TEMPLATE):
                body = template.format(doc=_doc(rng), month=rng.choice(MONTHS), amt=f"{rng.randint(500, 900000):,}",
                                       gstin=_gstin(rng), days=rng.randint(2, 10))
                text = _maybe_noise(rng.choice(OPENERS) + body + rng.choice(CLOSERS), rng)
                rows.append({"text": text, "label": label, "split": split, "template_id": f"{label}_{t_idx}"})
    rng.shuffle(rows)
    return rows


def main() -> None:
    out = Path(__file__).parent / "vendor_replies.csv"
    rows = generate()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "split", "template_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
