from typing import Generator, Callable, Iterable
import os

from flx.data.dataset import Identifier, IdentifierSet, DataLoader


def _get_subdirs_with_files(
    root_dir: str, extension: str | Iterable[str]
) -> Generator[tuple[str, list[tuple[str, str]]], None, None]:
    """
    Yields tuple of relative path from root to subdirectory and list of (filename_without_ext, ext)
    for every subdirectory that contains files matching one of the extensions.
    """
    if isinstance(extension, str):
        extensions = {
            extension.lower()
            if extension.startswith(".")
            else "." + extension.lower()
        }
    else:
        extensions = {
            ext.lower() if ext.startswith(".") else "." + ext.lower()
            for ext in extension
        }

    for dir_path, _, files in os.walk(root_dir):
        matched_files = []
        for file in files:
            stem, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                matched_files.append((stem, ext))
        if not matched_files:
            continue
        relative_path_from_root = os.path.relpath(dir_path, root_dir)
        yield (relative_path_from_root, matched_files)


class FileIndex(DataLoader):
    def __init__(
        self,
        root_dir: str,
        file_extension: str | Iterable[str],
        id_from_path: Callable[[str, str], Identifier],
    ):
        """
        A DataLoader for filepaths.

        Discovers all files with the given extension(s) in `root_dir` and its subdirectories and
        handles mapping between relative paths and ids. For each path relative to `root_dir` there
        must be a unique id and vice versa.

        file_extension: e.g. ".png" or (".tif", ".bmp", ".jpg", ".png")
        id_from_path: function that maps a relative path to an id.
            E.g. if the file path is "~/dataset_dir/person005/finger08/file000.png"
            and the root_dir is "~/dataset_dir", then the function will receive
            ("person005/finger08", "file000") as arguments.
            The id could be Identifier(subject=6*100 + 7, finger=7, sample=0)

        Folder structure:
        root_dir/
            subdir/
                ...
                    subdir/
                        file.extension
                        file.extension
                        ...
                    subdir/
                        ...
        """
        self._root_dir: str = os.path.normpath(os.path.abspath(root_dir))

        if file_extension is None:
            self._file_extension: str | tuple[str, ...] = (
                ".tif",
                ".tiff",
                ".bmp",
                ".jpg",
                ".jpeg",
                ".png",
            )
            extensions = list(self._file_extension)
        elif isinstance(file_extension, str):
            self._file_extension: str | tuple[str, ...] = (
                file_extension
                if file_extension.startswith(".")
                else "." + file_extension
            )
            extensions = [self._file_extension]
        else:
            self._file_extension = tuple(
                ext if ext.startswith(".") else "." + ext
                for ext in file_extension
            )
            extensions = list(self._file_extension)

        # Tuple is (subdir, filename) where subdir is relative to root_dir
        self._id_to_path_components: dict[Identifier, tuple[str, str]] = {}
        self._id_to_extension: dict[Identifier, str] = {}

        for subdir, files in _get_subdirs_with_files(
            self._root_dir, extensions
        ):
            for stem, ext in files:
                try:
                    identifier = id_from_path(subdir, stem)
                except (ValueError, IndexError):
                    identifier = None
                if identifier is None:
                    continue
                self._id_to_path_components[identifier] = (subdir, stem)
                self._id_to_extension[identifier] = ext

        if len(self._id_to_path_components) == 0:
            print(
                f"FileDataset with root_dir '{self._root_dir}' is empty. No files with extension {self._file_extension} found."
            )
        self._ids = IdentifierSet(list(self._id_to_path_components.keys()))

    @property
    def ids(self) -> IdentifierSet:
        return self._ids

    def get(self, identifier: Identifier) -> str:
        subdir, file = self._id_to_path_components[identifier]
        ext = self._id_to_extension.get(
            identifier,
            self._file_extension
            if isinstance(self._file_extension, str)
            else "",
        )
        relpath = file if subdir == "." else os.path.join(subdir, file)
        path = os.path.join(self._root_dir, relpath + ext)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File in index not found: {path}")
        return path
