# Finalize Field Inventory

This document enumerates the context fields referenced by finalize-stage letter templates. Each table lists the fields required for a given `action_tag`, along with the data source and owning subsystem.

## dispute
| Field | Source | Owner |
|-------|--------|-------|
| accounts | Planner/outcome history | Planner |
| bureau_address | External lookup | External data |
| bureau_name | Tri-merge evidence | Credit report ingestion |
| client_address_lines | User-supplied PII | Client intake |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| client_name | User-supplied PII | Client intake |
| closing_paragraph | Planner/outcome history | Planner |
| date | Planner/outcome history | Planner |
| inquiries | Tri-merge evidence | Credit report ingestion |
| is_identity_theft | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| opening_paragraph | Planner/outcome history | Planner |

## goodwill
| Field | Source | Owner |
|-------|--------|-------|
| account_history_good | Planner/outcome history | Planner |
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| months_since_last_late | Planner/outcome history | Planner |

## custom_letter
| Field | Source | Owner |
|-------|--------|-------|
| body_paragraph | User-supplied PII | Client intake |
| client_city | User-supplied PII | Client intake |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| client_name | User-supplied PII | Client intake |
| client_state | User-supplied PII | Client intake |
| client_street | User-supplied PII | Client intake |
| client_zip | User-supplied PII | Client intake |
| date | Planner/outcome history | Planner |
| greeting_line | User-supplied PII | Client intake |
| recipient_name | User-supplied PII | Client intake |
| supporting_docs | User-supplied PII | Client intake |

## instruction & paydown_first
| Field | Source | Owner |
|-------|--------|-------|
| client_name | User-supplied PII | Client intake |
| date | Planner/outcome history | Planner |
| logo_base64 | External lookup | External data |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| accounts_summary | Planner/outcome history | Planner |
| percent | Planner/outcome history | Planner |
| tips | Planner/outcome history | Planner |
| advisories | Planner/outcome history | Planner |

## fraud_dispute
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| is_identity_theft | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## debt_validation
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| collector_name | Tri-merge evidence | Credit report ingestion |
| days_since_first_contact | Planner/outcome history | Planner |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## pay_for_delete
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| collector_name | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| offer_terms | Planner/outcome history | Planner |

## mov
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| cra_last_result | Planner/outcome history | Planner |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| days_since_cra_result | Planner/outcome history | Planner |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## personal_info_correction
| Field | Source | Owner |
|-------|--------|-------|
| client_address_lines | User-supplied PII | Client intake |
| client_name | User-supplied PII | Client intake |
| date_of_birth | User-supplied PII | Client intake |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| ssn_last4 | User-supplied PII | Client intake |

## cease_and_desist
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| collector_name | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## direct_dispute
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| furnisher_address | External lookup | External data |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## bureau_dispute
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## inquiry_dispute
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| inquiry_creditor_name | Tri-merge evidence | Credit report ingestion |
| inquiry_date | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |

## medical_dispute
| Field | Source | Owner |
|-------|--------|-------|
| account_number_masked | Tri-merge evidence | Credit report ingestion |
| amount | Tri-merge evidence | Credit report ingestion |
| bureau | Tri-merge evidence | Credit report ingestion |
| client_context_sentence | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| creditor_name | Tri-merge evidence | Credit report ingestion |
| legal_safe_summary | Stage 2.5 canonical tags | Strategy/Stage 2.5 |
| medical_status | Tri-merge evidence | Credit report ingestion |
