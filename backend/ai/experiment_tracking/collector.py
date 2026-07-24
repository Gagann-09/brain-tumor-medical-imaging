import hashlib
import os


class ArtifactCollector:
    """
    Non-recursive directory scanner that identifies artifacts by absolute path.
    If the file exists in the cache, it recomputes the checksum; if it differs,
    it yields it as an updated artifact.
    """
    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        self.cache: dict[str, str] = {}  # maps absolute path to last seen checksum

    def _calculate_checksum(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def scan(self) -> list[tuple[str, str]]:
        """
        Scans the target directory non-recursively.
        Returns a list of (filepath, checksum) for new or updated files.
        """
        new_or_updated = []
        if not os.path.exists(self.target_dir):
            return new_or_updated

        with os.scandir(self.target_dir) as it:
            for entry in it:
                if entry.is_file():
                    abs_path = os.path.abspath(entry.path)
                    current_checksum = self._calculate_checksum(abs_path)

                    if abs_path not in self.cache or self.cache[abs_path] != current_checksum:
                        self.cache[abs_path] = current_checksum
                        new_or_updated.append((abs_path, current_checksum))

        return new_or_updated
