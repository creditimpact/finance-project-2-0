.PHONY: verify-note-style verify-validation

verify-note-style:
	python -m devtools.verify_note_style

verify-validation:
	python -m devtools.verify_validation
