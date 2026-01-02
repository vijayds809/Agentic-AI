import re
from typing import Callable, Any, Dict

def fetch_invoice_status(invoice_number: str) -> dict:
    if invoice_number == "TAUK144863":
        return {
            "Invoice Number": "TAUK144863",
            "Vendor Name": "Acme Supplies",
            "Status": "Paid",
            "Payment Date": "2025-10-15",
            "Net Amount": "₹25,000",
            "Payment Reference": "00104255",
        }
    return {}

def fetch_remittance(payment_reference: str) -> list:
    if payment_reference == "00104255":
        return [{
            "Vendor Name": "Acme Supplies",
            "Invoice Number": "TAUK144863",
            "Gross Amount": "₹25,000",
            "Payment Reference": "00104255",
        }]
    return []

class LocalTool:
    def __init__(self, name: str, func: Callable[..., Any]):
        self.func = func

    async def ainvoke(self, params: Dict[str, Any]):
        if "InvoiceNumber" in params:
            return self.func(params["InvoiceNumber"])
        if "PaymentReference" in params:
            return self.func(params["PaymentReference"])
        if "query" in params:
            q = params["query"]
            inv = re.search(r"TAUK\d+", q)
            pay = re.search(r"\d{6,12}", q)
            if inv:
                return self.func(inv.group())
            if pay:
                return self.func(pay.group())
        return self.func(None)
