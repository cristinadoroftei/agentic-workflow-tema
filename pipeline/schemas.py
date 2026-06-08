from pydantic import BaseModel, Field


class Invoice(BaseModel):
    """Schema for extracting invoice data."""

    numar: str = Field(description="Invoice number (e.g. FV-2024-001)")
    data: str = Field(description="Issue date (format: YYYY-MM-DD)")
    furnizor: str = Field(description="Supplier/vendor name")
    client: str = Field(description="Client/buyer name")
    total: float = Field(description="Total amount to pay (including VAT)")
    moneda: str = Field(default="RON", description="Currency (RON, EUR, etc.)")
    produse: list[str] = Field(
        default=[],
        description="List of products or services on the invoice",
    )


class Contract(BaseModel):
    """Schema for extracting contract data."""

    numar: str = Field(description="Contract number (e.g. CC-2024-008)")
    data_incheierii: str = Field(description="Signing date (format: YYYY-MM-DD)")
    prestator: str = Field(description="Service provider / contractor name")
    beneficiar: str = Field(description="Client / beneficiary name")
    obiect: str = Field(description="Short summary of the contract's object/scope")
    valoare_totala: float = Field(description="Total contract value including VAT")
    moneda: str = Field(default="RON", description="Currency (RON, EUR, etc.)")
    durata: str = Field(description="Contract duration (e.g. '6 luni')")
    data_inceput: str = Field(description="Start date (format: YYYY-MM-DD)")
    data_sfarsit: str = Field(description="End date (format: YYYY-MM-DD)")
    clauze_importante: list[str] = Field(
        default=[],
        description="Key clauses: confidentiality, penalties, termination conditions",
    )
