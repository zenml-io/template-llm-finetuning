# {% include 'template/license_header' %}

import os
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, ClassVar, Tuple, Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class DirectoryMaterializer(BaseMaterializer):
    """Materializer to store local directories in the artifact store."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (Path,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Any:
        """Copy the artifact files to a local temp directory.

        Args:
            data_type: Unused.

        Returns:
            Path to the local directory that contains the artifact files.
        """
        directory = mkdtemp(prefix="zenml-artifact")
        self._copy_directory(src=self.uri, dst=directory)
        return Path(directory)

    def save(self, data: Any) -> None:
        """Store the directory in the artifact store.

        Args:
            data: Path to a local directory to store.
        """
        assert isinstance(data, Path)
        self._copy_directory(src=str(data), dst=self.uri)

    @staticmethod
    def _copy_directory(src: str, dst: str) -> None:
        """Recursively copy a directory.

        Args:
            src: The directory to copy.
            dst: Where to copy the directory to.
        """
        for src_dir, _, files in fileio.walk(src):
            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
            fileio.makedirs(dst_dir)

            for file in files:
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(dst_dir, file)
                fileio.copy(src_file, dst_file)
