"""A small, fixed set of documents and questions with known answers.

Used to check retrieval end to end: each question must retrieve a passage
from its expected document. Keep it small and unambiguous.
"""

DOCUMENTS: dict[str, str] = {
    "finance.txt": (
        "Quarterly financial report. In the third quarter, revenue grew by twelve "
        "percent to four million dollars, driven by strong customer demand. "
        "Operating costs fell slightly because of cheaper cloud hosting."
    ),
    "handbook.txt": (
        "Employee handbook. Every employee receives twenty five paid vacation days "
        "per year. Vacation requests must be approved by a manager two weeks ahead."
    ),
    "office.txt": (
        "Office notes. The office cat is named Biscuit. Biscuit sleeps on the "
        "windowsill and is fed twice a day by the reception team."
    ),
}

QUESTIONS: list[tuple[str, str]] = [
    ("How much did revenue grow in the third quarter?", "finance.txt"),
    ("How many paid vacation days does an employee get per year?", "handbook.txt"),
    ("What is the name of the office cat?", "office.txt"),
]
