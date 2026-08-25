.PHONY: validate paper clean checksums repository-manifest evidence-bundle release-manifest

validate:
	python scripts/validate_repository.py --strict

paper:
	cd paper && latexmk -pdf main.tex

clean:
	cd paper && latexmk -C

checksums:
	python scripts/generate_checksums.py

repository-manifest:
	python scripts/generate_repository_manifest.py

evidence-bundle:
	python scripts/create_public_evidence_bundle.py

release-manifest:
	python scripts/generate_release_manifest.py
