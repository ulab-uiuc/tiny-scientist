import os.path as osp
from typing import Any, Dict, Optional

from rich import print

from .llm import get_response_from_llm


class BibManager:
    def __init__(self, model: str, client: Any) -> None:
        self.model = model
        self.client = client

    def _update_bib_cite(
        self, references: Dict[str, Any], dest_template_dir: str, template: str
    ) -> None:
        if template == "acl":
            bib_path = osp.join(dest_template_dir, "custom.bib")
        if template == "iclr":
            # you should create a custom.bib file in the iclr folder
            bib_path = osp.join(dest_template_dir, "custom.bib")

        bib_entries = []
        for meta in references.values():
            bibtex = meta.get("bibtex", "").strip()
            if bibtex:
                bib_entries.append(bibtex)

        if not bib_entries:
            print("No BibTeX entries to write.")
            return

        # Write all entries to the bib file
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(bib_entries))

        print(f"custom.bib created with {len(bib_entries)} entries.")

    def _get_bibtex_for_key(self, key: str) -> Optional[str]:
        prompt = f"Provide the bibtex entry for the paper with citation key '{key}'. Output only the bibtex entry."
        try:
            result = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model,
                system_message="You are an expert in academic citations. Please provide a valid bibtex entry.",
            )

            if isinstance(result, tuple):
                bibtex_entry = result[0]
            else:
                bibtex_entry = result

            if (
                isinstance(bibtex_entry, str)
                and "@" in bibtex_entry
                and key in bibtex_entry
            ):
                return bibtex_entry.strip()
            else:
                print(f"Invalid bibtex returned for key: {key}")
                return None

        except Exception as e:
            print(f"Error fetching bibtex for key '{key}': {e}")
            return None
