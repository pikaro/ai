from __future__ import annotations

import ast
import shlex
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).parents[1]
SERVICE_SOURCE_ROOTS = {
    'assistant': REPOSITORY / 'assistant' / 'src',
    'stt': REPOSITORY / 'stt' / 'src',
    'tts': REPOSITORY / 'tts' / 'src',
}
TRANSPORT_MODULES = frozenset(
    {
        'assistant.src.api',
        'assistant.src.configuration_api',
        'assistant.src.dashboard',
        'assistant.src.dependencies',
        'assistant.src.realtime',
        'assistant.src.schemas',
        'stt.src.api',
        'stt.src.schemas',
        'tts.src.api',
        'tts.src.schemas',
    },
)
FRAMEWORK_IMPORTS = frozenset({'fastapi', 'starlette'})


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def module_name(service: str, path: Path) -> str:
    relative = path.relative_to(SERVICE_SOURCE_ROOTS[service]).with_suffix('')
    return '.'.join((service, 'src', *relative.parts))


class ServiceDependencyArchitectureTest(unittest.TestCase):
    def test_web_framework_is_confined_to_transport_modules(self) -> None:
        problems: list[str] = []
        for service, source_root in SERVICE_SOURCE_ROOTS.items():
            for path in source_root.rglob('*.py'):
                current_module = module_name(service, path)
                problems.extend(
                    f'{path.relative_to(REPOSITORY)} imports {imported}'
                    for imported in imported_modules(path)
                    if (
                        imported.split('.', maxsplit=1)[0] in FRAMEWORK_IMPORTS
                        and current_module not in TRANSPORT_MODULES
                    )
                )

        self.assertEqual(problems, [], f'framework imports below API boundary: {problems}')

    def test_processing_modules_do_not_depend_on_transports(self) -> None:
        problems: list[str] = []
        for service, source_root in SERVICE_SOURCE_ROOTS.items():
            for path in source_root.rglob('*.py'):
                current_module = module_name(service, path)
                if current_module in TRANSPORT_MODULES or path.name == 'main.py':
                    continue
                problems.extend(
                    f'{path.relative_to(REPOSITORY)} imports {imported}'
                    for imported in imported_modules(path)
                    if imported in TRANSPORT_MODULES
                )

        self.assertEqual(problems, [], f'processing-to-transport dependencies: {problems}')

    def test_services_share_contracts_without_cross_importing_implementations(self) -> None:
        problems: list[str] = []
        service_names = frozenset(SERVICE_SOURCE_ROOTS)
        for service, source_root in SERVICE_SOURCE_ROOTS.items():
            foreign_services = service_names - {service}
            for path in source_root.rglob('*.py'):
                problems.extend(
                    f'{path.relative_to(REPOSITORY)} imports {imported}'
                    for imported in imported_modules(path)
                    if imported.split('.', maxsplit=1)[0] in foreign_services
                )

        self.assertEqual(problems, [], f'cross-service implementation imports: {problems}')

    def test_shared_contracts_do_not_depend_on_service_implementations(self) -> None:
        problems: list[str] = []
        forbidden_roots = frozenset(SERVICE_SOURCE_ROOTS) | FRAMEWORK_IMPORTS
        for path in (REPOSITORY / 'service_contracts').rglob('*.py'):
            problems.extend(
                f'{path.relative_to(REPOSITORY)} imports {imported}'
                for imported in imported_modules(path)
                if imported.split('.', maxsplit=1)[0] in forbidden_roots
            )

        self.assertEqual(problems, [], f'shared-contract dependencies: {problems}')

    def test_entrypoints_only_compose_uvicorn_and_api(self) -> None:
        problems: list[str] = []
        for service, source_root in SERVICE_SOURCE_ROOTS.items():
            path = source_root / 'main.py'
            allowed = {'__future__', 'uvicorn', f'{service}.src.api'}
            unexpected = set(imported_modules(path)) - allowed
            if unexpected:
                problems.append(
                    f'{path.relative_to(REPOSITORY)} imports {sorted(unexpected)}',
                )

        self.assertEqual(problems, [], f'entrypoint dependencies: {problems}')


class DockerBuildContextTest(unittest.TestCase):
    def test_local_copy_sources_exist_and_are_allowlisted(self) -> None:  # noqa: C901
        allowlist = {
            line
            for line in (REPOSITORY / '.dockerignore').read_text().splitlines()
            if line.startswith('!')
        }
        problems: list[str] = []

        for service in SERVICE_SOURCE_ROOTS:
            dockerfile = REPOSITORY / service / 'Dockerfile'
            for line_number, line in enumerate(dockerfile.read_text().splitlines(), start=1):
                stripped = line.strip()
                if not stripped.startswith('COPY ') or '--from=' in stripped:
                    continue
                tokens = [
                    token for token in shlex.split(stripped)[1:] if not token.startswith('--')
                ]
                for source in tokens[:-1]:
                    normalized = source.removeprefix('./').rstrip('/')
                    source_path = REPOSITORY / normalized
                    location = f'{dockerfile.relative_to(REPOSITORY)}:{line_number}'
                    if not source_path.exists():
                        problems.append(f'{location} copies missing {source}')
                        continue
                    required_patterns = (
                        {f'!{normalized}/', f'!{normalized}/**'}
                        if source_path.is_dir()
                        else {f'!{normalized}'}
                    )
                    missing = required_patterns - allowlist
                    if missing:
                        problems.append(
                            f'{location} copies ignored {source}; missing {sorted(missing)}',
                        )

        self.assertEqual(problems, [], f'invalid Docker COPY sources: {problems}')
