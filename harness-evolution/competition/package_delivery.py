#!/usr/bin/env python3
"""Assemble a whitelisted, hash-checked source and application delivery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
COMPETITION = 'harness-evolution/competition'
KEY_PATTERN = re.compile(rb'\bsk-[A-Za-z0-9_-]{32,100}\b')
INVENTORY = 'bundle-manifest.json'
AGH_COMMIT = '7e9aaa163b1a46eb6fd632ea0bf267202449cd0a'
AGH_CANDIDATE_COMMIT = '271487a50514cbd4208309e884c6c9ef3e3c31af'
AGH_CANDIDATE_ARCHIVE_SHA = '510bfb0d1868823ff93b18a9ea8432ed7106e8eb0e88ce8e319d74f5d53508d7'
AGH_ARCHIVE_SHA = 'c30914af3e9f8cadbea363cf479f69f46998a29fbae903a8e07a2d6f2a2f279f'
AGH_NEXT_COMMIT = '8195f2debdff45e9702d4187b2eac698482c8585'
AGH_NEXT_ARCHIVE_SHA = 'ee0995854cf1357aa25d503d40a5ceb39dd32f9fdc5c52848457923b1739f728'
AGH_LATEST_COMMIT = '79329a1e36b9e2c9785bae1e29d5b788d7149e26'
AGH_LATEST_ARCHIVE_SHA = '258c902ecbba7b2997b22806c0a90c6593bd7e0913223d23fdf9912b3970c86c'
AGH_STRUCTURED_COMMIT = 'c2de21af179b3fb24f1a67147ba871322f94493a'
AGH_STRUCTURED_ARCHIVE_SHA = 'e970fdce3ccaa3ac9bcd23d3d324267bd729e3758c7d28fd07491d9ad0ac1469'
AGH_BASELINE_COMMIT = '54b05f2a959b5f6328ac2bf572fbd981bf567094'
AGH_BASELINE_ARCHIVE_SHA = '68c9d5d6943f48e39cdb64cc41ad85548165a4b7c3f2e46b7f5ae917e4401c58'
AGH_INLINE_COMMIT = '9253a7d641fb87a9518866e8ed8493d31292a16d'
AGH_INLINE_ARCHIVE_SHA = 'f5a4f74d9a95374294347a2ef84ec9f4413605a99d7afd756ce6af180ca09ae8'
CURRENT_SOURCES = {
    'g009': {
        'commit': '3169b052487e3c6b2ed660edde5105568b7711dd',
        'parent': AGH_INLINE_COMMIT,
        'archive_sha256': '1800bdb7d73a94822254e59a4d2ea527e044a620d823fd3711ad267d69ff9df3',
        'patch': 'harness-evolution/codex-rsi/source-patches/G009-native-request-budget-guidance.patch',
        'patch_sha256': 'bbec3238a5ca0dd9d9391dee5881fc74e6a495d8efac2a77bd63a76d5d074f58',
        'scope': 'Native budget guidance; closed V21 original gates did not promote this source.',
    },
    'g010': {
        'commit': 'c8b633a4f646039a7ef1b3b874a79ea194b5cc83',
        'parent': '3169b052487e3c6b2ed660edde5105568b7711dd',
        'archive_sha256': 'fa43471b8e65e786359b9ed27ffc4c28b536789b224d87f45c260e2e8e631e9e',
        'patch': 'harness-evolution/codex-rsi/source-patches/G010-structured-correctness-feedback.patch',
        'patch_sha256': 'f5e0e13ed66070da20bd9b5986372b14f98e71ce2d59266d73aa6f9c53c9c95a',
        'scope': 'Validated correctness feedback; the single closed V22 DEV ablation is not source promotion.',
    },
    'g011': {
        'commit': 'd0ce85bec9fa73702ae7243485265f7f3281634e',
        'parent': 'c8b633a4f646039a7ef1b3b874a79ea194b5cc83',
        'archive_sha256': '15cf17e1ef5482900d3736070e18a0875212083e6fd6ef283bd33f7900a86165',
        'patch': 'harness-evolution/codex-rsi/source-patches/G011-static-compiler-repair-feedback.patch',
        'patch_sha256': '7fa6aa12c79b35aa922d7d98710f1f306149e17827bc41742713e4f45d74c230',
        'scope': 'Static verified compiler-failure recovery; closed V24 failed correctness and performance gates, no promotion.',
    },
    'g012': {
        'commit': '6818acb68cfcb7bdffe4db060a6a89f40f678a3a',
        'parent': 'c8b633a4f646039a7ef1b3b874a79ea194b5cc83',
        'archive_sha256': '4d596cda2cb278ff5bf1a6fcd92a638dbedfc391e5b55a39ec7135cb66c5a8ee',
        'patch': 'harness-evolution/codex-rsi/source-patches/G012-correct-unpromoted-measurement-feedback.patch',
        'patch_sha256': '6f469d66b7041db39ae56f7b550639fd1f13b9502430ca92b8551f8a4a64d929',
        'scope': 'Source-bound correct/unpromoted DEV guidance; closed V25 failed correctness, confidence and cost gates, no promotion.',
    },
    'g013': {
        'commit': '261086362deddcc740370d1c52727b59e7a25704',
        'parent': 'c8b633a4f646039a7ef1b3b874a79ea194b5cc83',
        'archive_sha256': '6abc3782141fc7c5ebdd8dc266fdf2450c7c0641768b02c17df34c57b45ef94a',
        'patch': 'harness-evolution/codex-rsi/source-patches/G013-bounded-pending-query.patch',
        'patch_sha256': '0a0672a8773a8ad9f52fe0dbdc9e232fb9233d442b1862c06bc940443514d35d',
        'source_file_count': 3629,
        'scope': 'One bounded pending DEV job continuation; closed V26 failed full evaluation, confidence and GPU cost gates, no promotion.',
    },
    'g014': {
        'commit': 'e1dffe9a8eca77cb1fdd81c1a8b5842c8c36370e',
        'parent': '261086362deddcc740370d1c52727b59e7a25704',
        'archive_sha256': '8c402576ffe1ba13fcecfe826da6a87fac92b2d32b872746a8e0c9e32508f342',
        'patch': 'harness-evolution/codex-rsi/source-patches/G014-running-only-pending-query.patch',
        'patch_sha256': 'b10e4907d4ef9734edcb1e5595d39d356117e14278c0345ea287ae9f59799512',
        'source_file_count': 3629,
        'scope': 'Running-only bounded continuation; queued first results preserved, actual queued provider tradeoff retained. CPU qualification only, paid comparison pending.',
    },
    'g017': {
        'commit': 'a3a80e7a856789937f0225a5bc30e16e02143ca5',
        'parent': 'e1dffe9a8eca77cb1fdd81c1a8b5842c8c36370e',
        'archive_sha256': 'db6266c3bfb25c1771e2bd6ed233cf2735f9938d7a2448a6739040aafc6056ad',
        'patch': 'harness-evolution/codex-rsi/source-patches/G017-repeated-source-copy-feedback.patch',
        'patch_sha256': 'caee46df85eddf85f52a057ef6618cfe92673443af2213f37b1949e9fea0fbcb',
        'source_file_count': 3630,
        'scope': 'Bounded same-source write feedback; actual writes preserved, 67 original SDK CPU bodies qualified. Paid V30 ablation and source benefit pending.',
    },
}
CURRENT_REPORTS = {
    'harness-evolution/codex-rsi/reports/' + name: digest for name, digest in {
        'g008-v19-original-series-full-eight.json': '21ce4a2756d9d8947569c734494699ed2b715b685ca7b32f9c03ccf0df40fecb',
        'G009-source-runtime-actual-SDK-CPU-qualification-r1.json': '12fe9d48b5f6c919ce3c667434811a8cfcd9fde4af872429ffba46dca94115ce',
        'g009-v21-original-series-full-eight-r1.json': '46d4070feab0a1160278ab6cdcc5eb9f2f160642ab71561e0cb8c70c75152677',
        'g009-v21-actual-closed-DEV-diagnosis-r1.json': '8aa27135fbfb35bbd1b69c240c9ed56306c3efa81b9e6398496d064e429761a0',
        'g010-structured-correctness-feedback-actual-CPU-qualification-r1.json': '5d7858b708ba5626ce5119fc34b86ba74da8454bad9ddfe09eba6f61db2d4a77',
        'g010-phase22-original-two-arm-ablation-r1.json': 'b5b93d02daf4f5e7b6a765c3cedf63c772c454eee1983495aeb61f0bee2d8d54',
        'g010-phase22-actual-closed-DEV-diagnosis-and-exact-history-r1.json': '56e9e8a78bf977633a74e6772bf5cadb8e37d35d15bff41bb682fa003e00552c',
        'g010-v23-original-full-eight-r1.json': '9f99dd1a6a13f550579aa849327a9ffca8066e126fb6df0a01f35cc192a07e17',
        'g010-v23-closed-DEV-diagnosis-exact-history-r1.json': 'a22757d052ecda2b68f3eba8ad6b2f5b1c8ab0886f0cc40014f168cdaa7ccc33',
        'g011-static-compiler-feedback-actual-CPU-qualification-r1.json': 'cfc1d284d727b06c63f146673a3747693566394447badd6eacf0963f7668c46a',
        'g011-v24-original-two-arm-ablation-r1.json': '2ba0324de02dbe57f55e01af52d6383cbc408c57b3980ebf520cde995a63d20d',
        'g011-v24-closed-DEV-diagnosis-exact-history-r1.json': 'f07275aeecd69a9c5d6233d69227d20c07be1cf45be07e002642f970cb2e0e9e',
        'g012-unpromoted-feedback-actual-CPU-qualification-r1.json': '5aaeb0cc1577fc8ba44a80d1dd94ff9c846a34aaacef434641d437f6916d757c',
        'phase25-pre-model-actual-history-listener-RPC-wire-CPU-qualification-r1.json': '2d467e12cf4d2b4b6d2cd7736c41383bde53ee0f33a2fc4f84e3809639baacb9',
        'g012-v25-original-two-arm-ablation-r1.json': 'fad23f29ddf51364c71f2abb537e1794e381c717517d7340750231101e6f75f4',
        'g012-v25-closed-DEV-diagnosis-exact-history-r1.json': '51c424e00ec5fff2ba7f7167dd4f5e924b64f04e7b192b00dfd42b1fde8dba4a',
        'g013-bounded-pending-query-actual-CPU-qualification-r1.json': 'a4ecd70b0b573b6350a2fd97edcb528c9e7f548353544c132d972de1c0026ec2',
        'phase26-pre-model-actual-history-listener-RPC-wire-CPU-qualification-r1.json': '7cb9a515fc8e4b6f8e1dd5b132f53a5be69a3cc1492633c572408aa8d5a9e7cb',
        'g013-v26-original-two-arm-ablation-r1.json': '6374d6dd705b9d4156127d67d1db28fc856cbc3357664bcf070f4a81abd4f0be',
        'g013-v26-closed-DEV-diagnosis-exact-history-r1.json': '5dbe48ed1161006d9a9b9adee81bb4348902d41c149323b8169ba36ab7688de8',
        'g014-running-only-pending-query-actual-CPU-qualification-r1.json': '29f74e272f2d78871b107b15299ea07140f0bd2c29fe480841b45e7fc2327a40',
        'g017-source-independent-build-qualification-r1.json': 'fa1689844767ac12c9f87172a4b2b5582cb06607b12080cca5db88469bffcc19',
        'g017-source-copy-actual-SDK-qualification-r1.json': '7d6f36eb9784172bba2d28b1892f5a63a5e536b7e33a1d9568c168d7344df6c9',
        'g016-v29-original-two-arm-public-all40-r1.json': '90a88ec3408001a1ad31104a01683aded294bea51df71625ee73fe1c852ae034',
        'g016-v29-closed-DEV-diagnosis-exact-history-r1.json': '2844eb75ede40fa6ab72da6c98659988092485e307669e6cf9055390c40656c0',
        'phase30-source-copy-actual-RPC-model-ready-GPU-pending-r1.json': '5ae2104cb6441ade969110d07d3fda7a42844eee291ae6727ca12a8d3c0725b4',
        'post-V26-application-original-history-and-RPC-actual-CPU-qualification-r1.json': 'c5f9c923d2004a8954187941f11fb56f21eb18c2d1256872739c5950e462b7c9',
        'citation-graph-split-compiled-actual-two-device-GPU-post-V19-r1.json': '67eccbadda2471354b1d6f286e05bc4ba68ae9cb25716ee5f54a2227fae2b25b',
        'citation-graph-recorded-presentation-actual-browser-qualification-r1.json': '407c536dacd516e09f87e1c4a7b815fd6c0e8619d8ecad0787f1b5d77d31e8bb',
        'citation-graph-independent-reproduction-actual-CPU-qualification-r1.json': '46943395e264f8da7d5e0fccb9233fb7a55508d58001c429ef293d7f31c30005',
        'citation-graph-complete-sample-export-actual-CPU-verification-r1.json': '30c6129fc47e09454fbe71c7ad46368d0f51deeab85420ccc259071f0c1c866f',
        'CSR-original-protocol-two-device-post-V19-r1.json': '4efbd6d6355f30ca9ab2997ed4fa339c7cd91c29d1895afa9577e04ab559adae',
    }.items()
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(name):
    path = PurePosixPath(str(name))
    if path.is_absolute() or not path.parts or any(p in {'..', '.'} for p in path.parts) or '\\' in str(name):
        raise ValueError('unsafe delivery entry path')
    return path


def inspect_bytes(data, name):
    if KEY_PATTERN.search(data):
        raise ValueError('possible credential in delivery entry: ' + str(name))


def safe_copy(source_root, name, target_root):
    name = safe_name(name)
    source_root = Path(source_root)
    source = source_root.joinpath(*name.parts)
    if not source.is_file() or any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError('delivery source must be a regular file without symlinks')
    if not source.resolve().is_relative_to(source_root.resolve()):
        raise ValueError('delivery source escapes root')
    data = source.read_bytes()
    inspect_bytes(data, name)
    target = Path(target_root).joinpath(*name.parts)
    if target.exists() or any(p.is_symlink() for p in target.parents):
        raise ValueError('delivery destination already exists or has symlink ancestors')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    if sha(target) != sha(source):
        raise ValueError('delivery copy changed source bytes')
    return target


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def portable_operators(source, target):
    source, target = Path(source), Path(target)
    original = source / 'manifest.json'
    data = json.loads(original.read_bytes())
    if data.get('kind') != 'source-bound-verified-kernel-delivery':
        raise ValueError('operator provenance kind differs')
    rows = []
    seen = set()
    for row in data['operators']:
        if row['task_id'] in seen or not row['verify_job_id']:
            raise ValueError('operator identity missing or duplicated')
        seen.add(row['task_id'])
        kernel = safe_copy(source, row['source'], target)
        if sha(kernel) != row['source_file_sha256']:
            raise ValueError('operator source provenance changed')
        rows.append({k: row[k] for k in ('task_id', 'source', 'source_file_sha256', 'candidate_id', 'verify_job_id')})
    write_json(target / 'manifest.json', {'kind': data['kind'], 'operators': rows})
    receipt = {'original_manifest_sha256': sha(original), 'portable_manifest_sha256': sha(target / 'manifest.json'),
        'executed_kernel_bytes_unchanged': True, 'original_verification_identities_preserved': True,
        'private_reference_paths_omitted': True, 'new_manifest_is_not_original_measurement_receipt': True}
    write_json(target / 'portable-provenance.json', receipt)
    return receipt


def write_inventory(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('delivery inventory contains a symlink')
        if path.is_file() and path.name != INVENTORY:
            name = path.relative_to(root).as_posix()
            safe_name(name)
            inspect_bytes(path.read_bytes(), name)
            rows.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    write_json(root / INVENTORY, {'schema_version': 1, 'files': rows,
        'manifest_self_hash_in_external_build_receipt': True, 'credentials_included': False})
    return verify_inventory(root)


def verify_inventory(root):
    root = Path(root)
    document = json.loads((root / INVENTORY).read_bytes())
    expected = set()
    for row in document['files']:
        name = safe_name(row['path']).as_posix()
        if name in expected or name == INVENTORY:
            raise ValueError('duplicate or invalid inventory entry')
        expected.add(name)
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('inventory source missing or symlinked')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('inventory source changed: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected | {INVENTORY} or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('unlisted or missing delivery files')
    return {'status': 'PASS', 'verified_file_count': len(expected), 'manifest_sha256': sha(root / INVENTORY)}


def check_agh_archive(archive):
    if sha(archive) != AGH_ARCHIVE_SHA:
        raise ValueError('frozen AGH source archive identity changed')
    with tarfile.open(archive, 'r:gz') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('AGH archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        if len(files) != 3626 or not {'LICENSE', 'NOTICE'}.issubset(files):
            raise ValueError('frozen AGH archive source/legal inventory changed')
    return files


def check_candidate_source(archive):
    if sha(archive) != AGH_CANDIDATE_ARCHIVE_SHA:
        raise ValueError('unbound AGH candidate archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('candidate archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        if len(files) != len(set(files)) or len(files) != 3626 or not {'LICENSE', 'NOTICE'}.issubset(files):
            raise ValueError('candidate source/legal inventory changed')
    return files


def check_next_source(archive):
    if sha(archive) != AGH_NEXT_ARCHIVE_SHA:
        raise ValueError('unbound AGH G005 source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('G005 archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        if len(files) != len(set(files)) or len(files) != 3627 or not {'LICENSE', 'NOTICE'}.issubset(files):
            raise ValueError('G005 source/legal inventory changed')
    return files


def check_latest_source(archive):
    if sha(archive) != AGH_LATEST_ARCHIVE_SHA:
        raise ValueError('unbound AGH G006 source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('G006 archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        if len(files) != len(set(files)) or len(files) != 3627 or not {'LICENSE', 'NOTICE'}.issubset(files):
            raise ValueError('G006 source/legal inventory changed')
    return files


def check_structured_source(archive):
    if sha(archive) != AGH_STRUCTURED_ARCHIVE_SHA:
        raise ValueError('unbound AGH G007 source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('G007 archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        required = {'LICENSE', 'NOTICE', 'packages/core/src/request/structured-json.ts'}
        if len(files) != len(set(files)) or len(files) != 3628 or not required.issubset(files):
            raise ValueError('G007 source/legal inventory changed')
    return files


def check_baseline_source(archive):
    if sha(archive) != AGH_BASELINE_ARCHIVE_SHA:
        raise ValueError('unbound original H0 source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('H0 archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        if len(files) != len(set(files)) or len(files) != 3625 or not {'LICENSE', 'NOTICE'}.issubset(files):
            raise ValueError('original H0 source/legal inventory changed')
    return files


def check_inline_source(archive):
    if sha(archive) != AGH_INLINE_ARCHIVE_SHA:
        raise ValueError('unbound AGH G008 source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('G008 archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        required = {'LICENSE', 'NOTICE', 'packages/core/src/request/structured-json.ts'}
        if len(files) != len(set(files)) or len(files) != 3628 or not required.issubset(files):
            raise ValueError('G008 source/legal inventory changed')
    return files


def check_current_source(archive, generation):
    binding = CURRENT_SOURCES[generation]
    if sha(archive) != binding['archive_sha256']:
        raise ValueError('unbound AGH ' + generation.upper() + ' source archive')
    with tarfile.open(archive, 'r:') as bundle:
        files = []
        for item in bundle.getmembers():
            safe_name(item.name)
            if not (item.isdir() or item.isfile()):
                raise ValueError('current AGH archive contains nonregular entries')
            if item.isfile():
                files.append(item.name)
                inspect_bytes(bundle.extractfile(item).read(), item.name)
        required = {'LICENSE', 'NOTICE', 'packages/core/src/step/job-feedback.ts'}
        if generation in ('g013', 'g014', 'g017'):
            required.add('packages/core/src/step/pending-job-query.ts')
        if generation == 'g017':
            required.add('packages/core/src/step/source-copy-feedback.ts')
        if len(files) != len(set(files)) or len(files) != binding.get('source_file_count', 3628) or not required.issubset(files):
            raise ValueError('current AGH source/legal inventory changed')
    return files


def validate_current_sources(args):
    request = getattr(args, 'agh_request_budget_source', None)
    correctness = getattr(args, 'agh_correctness_source', None)
    compiler = getattr(args, 'agh_compiler_source', None)
    unpromoted = getattr(args, 'agh_unpromoted_source', None)
    pending = getattr(args, 'agh_pending_source', None)
    running_pending = getattr(args, 'agh_running_pending_source', None)
    source_copy = getattr(args, 'agh_source_copy_source', None)
    if source_copy and not running_pending:
        raise ValueError('G017 delivery requires its G014 engineering parent')
    if running_pending and not pending:
        raise ValueError('G014 delivery requires its G013 engineering parent')
    if pending and not correctness:
        raise ValueError('G013 delivery requires its G010 engineering parent')
    if unpromoted and not correctness:
        raise ValueError('G012 delivery requires its G010 engineering parent')
    if compiler and not correctness:
        raise ValueError('G011 delivery requires its G010 engineering parent')
    if correctness and not request:
        raise ValueError('G010 delivery requires its G009 engineering parent')
    if request and not (getattr(args, 'agh_inline_source', None) and getattr(args, 'agh_baseline_source', None)):
        raise ValueError('G009 delivery requires original H0 and its G008 engineering parent')
    for source, generation in ((request, 'g009'), (correctness, 'g010'), (compiler, 'g011'), (unpromoted, 'g012'), (pending, 'g013'), (running_pending, 'g014'), (source_copy, 'g017')):
        if source:
            check_current_source(source, generation)
            binding = CURRENT_SOURCES[generation]
            if sha(ROOT / binding['patch']) != binding['patch_sha256']:
                raise ValueError('current AGH patch identity changed')
    return request, correctness


def package_current_source(archive, generation, agh, repo=ROOT):
    files = check_current_source(archive, generation)
    binding = CURRENT_SOURCES[generation]
    repo = Path(repo)
    if sha(repo/binding['patch']) != binding['patch_sha256']:
        raise ValueError('current AGH patch identity changed')
    destination = Path(agh)/('candidate-' + generation)
    destination.mkdir()
    archive = Path(archive)
    safe_copy(archive.parent, archive.name, destination)
    (destination/archive.name).rename(destination/'source.tar')
    safe_copy(repo, binding['patch'], destination)
    nested_patch = destination/binding['patch']
    nested_patch.rename(destination/'source.patch')
    for parent in (nested_patch.parent, *nested_patch.parent.parents):
        if parent == destination:
            break
        parent.rmdir()
    with tarfile.open(archive, 'r:') as bundle:
        for name in ('LICENSE', 'NOTICE'):
            with (destination/name).open('xb') as stream:
                stream.write(bundle.extractfile(name).read())
    receipt = {'source_commit': binding['commit'], 'parent_source_commit': binding['parent'],
        'archive_sha256': binding['archive_sha256'], 'patch_sha256': binding['patch_sha256'],
        'tracked_source_files': len(files), 'original_source_bytes_unmodified': True,
        'author': 'Codex outer loop', 'model_authorship_verified': False,
        'formal_source_performance_promoted': False, 'RSI_proven': False,
        'incumbent_source_commit': AGH_BASELINE_COMMIT, 'scope': binding['scope']}
    write_json(destination/'provenance.json', receipt)
    return receipt


def current_source_template(base, generation, comparison, **metadata):
    binding = CURRENT_SOURCES[generation]
    return {**base, **metadata, 'source_commit': binding['commit'],
        'parent_source_commit': binding['parent'],
        'engineering_parent_source_commit': binding['parent'],
        'comparison_parent_source_commit': AGH_BASELINE_COMMIT if generation in ('g009', 'g010') else binding['parent'],
        'incumbent_source_commit': AGH_BASELINE_COMMIT,
        'comparison': comparison, 'formal_source_performance_promoted': False,
        'RSI_proven': False}


def package_current_reports(out, repo=ROOT):
    repo = Path(repo)
    for name, digest in CURRENT_REPORTS.items():
        path = repo/name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('current public report identity changed')
    for name in CURRENT_REPORTS:
        safe_copy(repo, name, out)


REPRODUCE = '''#!/usr/bin/env python3
"""Verify the portable source inventory and prepare/run an independent app."""
import argparse, importlib.util, json, os
from pathlib import Path
import subprocess, sys
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path)
p.add_argument('--application', choices=['sequence', 'cached', 'packed', 'pretrained', 'packed-variant', 'pretrained-precision', 'pretrained-welford', 'citation-graph', 'pretrained-aligned-head'], default='cached')
p.add_argument('--model-dir', type=Path)
p.add_argument('--archive', type=Path)
p.add_argument('--case', choices=['all', 'arxiv-full', 'arxiv-first-2048', 'boundary-fixture'], default='all')
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--execute', action='store_true')
a = p.parse_args()
spec = importlib.util.spec_from_file_location('portable_package', root / 'harness-evolution/competition/package_delivery.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print(json.dumps(module.verify_inventory(root)), flush=True)
if a.out is None: sys.exit(0)
out = a.out.resolve()
if out.is_relative_to(root): raise ValueError('reproduction output must be outside the immutable bundle')
if a.application == 'citation-graph':
    if a.archive is None: p.error('citation-graph requires --archive containing the original local arxiv.zip; no data is downloaded')
    command = [sys.executable, str(root / 'harness-evolution/competition/reproduce_citation_graph.py'),
               '--archive', str(a.archive), '--out', str(out), '--case', a.case, '--gpu', str(a.gpu)]
    if a.execute: command += ['--execute']
    sys.exit(subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}).returncode)
operator = root / 'harness-evolution/codex-rsi/operators/g001/manifest.json'
entry = {'cached': 'reproduce_cached_application.py', 'packed': 'reproduce_packed_application.py',
         'sequence': 'reproduce_application.py', 'pretrained': 'reproduce_pretrained_application.py',
         'packed-variant': 'reproduce_packed_gemm_variant.py', 'pretrained-precision': 'reproduce_pretrained_precision.py',
         'pretrained-welford': 'reproduce_pretrained_welford.py',
         'pretrained-aligned-head': 'reproduce_pretrained_aligned_head.py'}[a.application]
if a.application in ('pretrained', 'pretrained-precision', 'pretrained-welford', 'pretrained-aligned-head'):
    if a.model_dir is None: p.error('pretrained requires --model-dir containing the declared local GPT-2 weights')
    operator = root / 'harness-evolution/codex-rsi/operators/g000-v14-dev1/manifest.json'
command = [sys.executable, str(root / 'harness-evolution/competition' / entry), '--out', str(out), '--gpu', str(a.gpu),
           '--operator-manifest', str(operator), '--operator-manifest-sha256', module.sha(operator)]
if a.application in ('pretrained', 'pretrained-precision', 'pretrained-welford', 'pretrained-aligned-head'): command += ['--model-dir', str(a.model_dir)]
if a.application in ('cached', 'packed'):
    k12 = root / 'harness-evolution/codex-rsi/operators/g003-v13-dev1/manifest.json'
    command += ['--k12-manifest', str(k12), '--k12-manifest-sha256', module.sha(k12)]
if a.application == 'packed-variant':
    variant = root / 'harness-evolution/codex-rsi/operators/g000-v15-dev2/manifest.json'
    command += ['--variant-manifest', str(variant), '--variant-manifest-sha256', module.sha(variant)]
if a.execute: command += ['--execute']
sys.exit(subprocess.run(command, cwd=root, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}).returncode)
'''


README = '''# KernelEvolve 可复现作品包

硕博组工程预览。队名、报名和发布信息按团队要求暂留空；本地文件不是已发布或已提交证明。

先运行 `python reproduce.py` 核对全部文件SHA256。环境需Python3.11、PyTorch2.5.1、Triton3.1.0及NumPy1.26.4；已测CUDA为12.1，版本记录在requirements-reproduction.txt。原仓库requirements.lock与requirements-cuda.lock仅作为历史源码文件保留，与本次实测环境不同，不作为本次复现安装入口。

CPU计划：`python reproduce.py --application cached --out /tmp/my-cached-plan`；合并投影版使用 `--application packed`，完整序列版使用 `--application sequence`。计划不调用模型，不初始化GPU。缺少依赖、原kernel源文件或hash不一致会拒绝，不能视作复现完成。

团队在自己的干净CUDA设备上重新实测时，另选全新输出目录，加 `--gpu 0 --execute`。独立本地官方Queue记录编译、正确性和全部GPU生命周期；不读取凭据、不触碰生产账本。实际报告保留失败、强库基线及30轮全部原样本。旧KV缓存输入不改写。

原独立算子收益相对项目冻结库基线；十二类交付，包含形状弱收益。新增归一化和CSR来自原H0版本；CSR原验证未新增跨设备复验。缓存V2已有不同物理GPU的完整独立复测。合并投影版对普通FP16库2.17–2.47倍，对每轮最快六种库基线尚未全形状胜出。不能相乘独立算子比值或宣称预训练文本质量提升。

真实预训练GPT-2入口：`python reproduce.py --application pretrained --model-dir /your/local/gpt2 --out /tmp/gpt2-plan`。使用原snapshot607a30d，model.safetensors SHA须为248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707；权重不包含在ZIP且不会自动下载。仅CPU计划不能证明Triton/GPU收益，实际GPU结果以报告为准。原K02残差LayerNorm与全部124439808个预训练参数保持，编译库基线缺失不能判胜。

独立Welford数值修复入口：`python reproduce.py --application pretrained-welford --model-dir /your/local/gpt2 --out /tmp/welford-plan`。默认仅生成CPU计划，不初始化或枚举CUDA；保留原四组形状、预训练参数和局部/完整logits容差。显式`--execute`在团队自己的干净GPU上执行原900秒数值诊断，使用独立本地Queue和GPU锁，不访问作者生产账本或网关。CUDA扩展从包含的.cpp/.cu源文件本地编译，需要兼容的CUDA工具链；本包不分发作者.so或模型权重。该入口未计时，结果不代表AGH性能晋级；旧K02三组完整模型失败继续保留。

输出头布局原型入口：`python reproduce.py --application pretrained-aligned-head --model-dir /your/local/gpt2 --out /tmp/aligned-head-plan`。默认只生成离线CPU计划，GPU收益尚未实测；历史K02核验仅作来源记录，本原型保留原vanilla backbone及全部预训练参数。候选为零补47列至50304的NN布局，TN布局为padding-only控制；slice和contiguous复制在每次计时forward内。显式执行仍使用独立Queue、GPU锁、原900秒、四形状、原logits容差、20轮预热/30配对及两种强库；缺失强基线或数值失败不能判胜。权重与GPU执行结果均不包含在本包，本原型不证明AGH进化或RSI。

AGH源码修改由外层Codex完成，G001至G003完整冻结源码、LICENSE/NOTICE及三份原始patch在 `agh/`。自行解压 `agh/source.tar.gz`，依其README安装Node24和pnpm并构建。AGH使用mimo-v2.6-pro，五槽off/context32768；凭据由团队自行配置到私有环境。公开包无可计费网关配置。20路实验控制器源码在 `harness-evolution/codex-rsi/`，需要团队独立登记私有工作区/网关/预算及实际GPU锁，不能直接复用作者生产路径。应用复现与AGH20路主性能对照分别验收。

若包含 `agh/baseline-g000/source.tar`，它是原H0对照commit54b05f2的完整3625文件源码，另有独立SHA和LICENSE/NOTICE来源登记。归档没有修改原源码或性能指针；从归档重新构建Node运行包的独立证明另行记录，不能把源码库存核对当作新运行包验证。

便携manifest保留原kernel字节和验证身份，新SHA身份在portable-provenance.json中说明，不是原测量记录的替代。严格依赖原SHA的新GEMM保留g003和g000-v15-dev2原manifest；其中历史路径只作来源记录，CPU计划验证包内相对kernel，不访问生产账本。默认核验不意味着第二次GPU实测。

`application/`公开报告位于 `harness-evolution/competition/application/`；如包含 `demo/`，可双击index.html或把demo目录作为HTTP根目录。结构回放是已记录的原生工具事件，不是实时执行。

主实验三组开发对照及独立留出仍须满足完整20任务、正确率、环境、配对置信区间和成本门；本包未宣称RSI性能成立。第三方许可见THIRD_PARTY_NOTICES.md；参赛团队应保留原始署名、许可证和NOTICE。项目自有代码的发布许可尚未指定，本包不为作者擅自授予新许可。
'''


def build(args):
    out = args.out.resolve()
    if out.exists() or args.out.is_symlink():
        raise ValueError('delivery output must be fresh')
    check_agh_archive(args.agh_archive)
    baseline_source = getattr(args, 'agh_baseline_source', None)
    if baseline_source:
        check_baseline_source(baseline_source)
    candidate = getattr(args, 'agh_candidate_source', None)
    if candidate:
        check_candidate_source(candidate)
    next_source = getattr(args, 'agh_next_source', None)
    if next_source:
        if not candidate:
            raise ValueError('G005 delivery requires its G004 engineering parent')
        check_next_source(next_source)
    latest_source = getattr(args, 'agh_latest_source', None)
    if latest_source:
        if not next_source:
            raise ValueError('G006 delivery requires its G005 engineering parent')
        check_latest_source(latest_source)
    structured_source = getattr(args, 'agh_structured_source', None)
    if structured_source:
        if not latest_source:
            raise ValueError('G007 delivery requires its G006 engineering parent')
        check_structured_source(structured_source)
    inline_source = getattr(args, 'agh_inline_source', None)
    if inline_source:
        if not structured_source:
            raise ValueError('G008 delivery requires its G007 engineering parent')
        check_inline_source(inline_source)
    request_source, correctness_source = validate_current_sources(args)
    compiler_source = getattr(args, 'agh_compiler_source', None)
    unpromoted_source = getattr(args, 'agh_unpromoted_source', None)
    pending_source = getattr(args, 'agh_pending_source', None)
    running_pending_source = getattr(args, 'agh_running_pending_source', None)
    source_copy_source = getattr(args, 'agh_source_copy_source', None)
    out.mkdir(parents=True, mode=0o700)
    for directory, suffixes in [('kernel_lab', {'.py'}), ('tasks', {'.py', '.md', '.json'}),
                               (COMPETITION + '/application', {'.py', '.md', '.json', '.cpp', '.cu'}),
                               ('harness-evolution/codex-rsi', {'.py', '.mjs'}),
                               ('harness-evolution/agh-adapter', {'.mjs', '.md', '.json'}),
                               ('harness-evolution/kernel-workflow-plugin', {'.mjs', '.json'}),
                               ('harness-evolution/recursive-self-improvement', {'.py', '.mjs'})]:
        files = (ROOT / directory).glob('*') if directory == 'harness-evolution/codex-rsi' else (ROOT / directory).rglob('*')
        for source in sorted(files):
            if source.is_file() and source.suffix in suffixes and '__pycache__' not in source.parts:
                # Include source modules only in controller trees; public app
                # reports are already sanitized and are explicitly allowed.
                safe_copy(ROOT, source.relative_to(ROOT).as_posix(), out)
    for name in ['pyproject.toml', 'requirements.lock', 'requirements-cuda.lock',
                 COMPETITION + '/package_delivery.py', COMPETITION + '/reproduce_application.py',
                 COMPETITION + '/reproduce_cached_application.py', COMPETITION + '/export_application_report.py',
                 COMPETITION + '/reproduce_packed_application.py',
                 COMPETITION + '/reproduce_pretrained_application.py',
                 COMPETITION + '/reproduce_pretrained_aligned_head.py',
                 COMPETITION + '/reproduce_packed_gemm_variant.py', COMPETITION + '/reproduce_pretrained_precision.py',
                 COMPETITION + '/reproduce_pretrained_welford.py',
                 COMPETITION + '/reproduce_citation_graph.py',
                 COMPETITION + '/export_citation_graph_samples.py',
                 COMPETITION + '/build_citation_graph_demo.py',
                 COMPETITION + '/build_current_demo.py',
                 COMPETITION + '/export_cached_application.py', COMPETITION + '/export_packed_application.py',
                 COMPETITION + '/export_pretrained_application.py',
                 COMPETITION + '/export_packed_gemm_variant.py', COMPETITION + '/export_pretrained_precision.py',
                 COMPETITION + '/export_series_report.py', COMPETITION + '/export_series_report_v14.py', COMPETITION + '/export_series_report_v15.py', COMPETITION + '/export_series_report_v16.py',
                 COMPETITION + '/export_series_report_v17.py', COMPETITION + '/export_series_report_v18.py', COMPETITION + '/build_evidence.py',
                 COMPETITION + '/generate_materials.py', COMPETITION + '/record_demo_video.py',
                 COMPETITION + '/verify_demo.py', COMPETITION + '/extend_evidence.py']:
        safe_copy(ROOT, name, out)
    for source in sorted((ROOT / 'tests').glob('test_competition*.py')):
        safe_copy(ROOT, source.relative_to(ROOT).as_posix(), out)
    for name in ('g006-source-runtime-cpu-verification.json', 'g006-v16-control-cpu-qualification.json',
                 'application-live-budget-status-cpu-verification.json', 'hackathon-v15-eight-arm-summary.json',
                 'g006-v16-original-series-full-eight.json', 'g007-source-runtime-cpu-verification.json',
                 'g007-v17-control-cpu-qualification.json', 'pretrained-original-k02-GPU-numerical-diagnosis-v16.json',
                 'g007-v17-stopped-two-closed-six-not-started.json', 'g007-v18-exact-capacity-control-cpu-qualification.json',
                 'g007-series-demo-CPU-verification-v18.json', 'fused-welford-pretrained-CPU-verification-v18.json',
                 'fused-welford-independent-reproduction-CPU-verification.json',
                 'g007-v18-original-series-full-eight.json',
                 'pretrained-fused-Welford-GPU-numerical-diagnosis-v18.json',
                 'g008-source-bound-engineering-actual-verification.json',
                 'g008-v19-control-actual-CPU-qualification.json'):
        safe_copy(ROOT, 'harness-evolution/codex-rsi/reports/' + name, out)
    if request_source:
        package_current_reports(out)
    for generation in ['g001', 'g003-v13-dev1', 'g000-v14-dev1', 'g000-v15-dev2', 'g000-v18-dev2']:
        target = out / 'harness-evolution/codex-rsi/operators' / generation
        target.mkdir(parents=True)
        original = ROOT / 'harness-evolution/codex-rsi/operators' / generation
        if generation in ('g003-v13-dev1', 'g000-v15-dev2', 'g000-v18-dev2'):
            manifest = json.loads((original / 'manifest.json').read_bytes())
            safe_copy(original, 'manifest.json', target)
            for row in manifest['operators']:
                copied = safe_copy(original, row['source'], target)
                if sha(copied) != row['source_file_sha256']:
                    raise ValueError('original frozen variant kernel bytes changed')
            write_json(target / 'portable-provenance.json', {'original_manifest_sha256': sha(original / 'manifest.json'),
                'manifest_bytes_unchanged': True, 'kernel_bytes_unchanged': True,
                'historical_paths_are_reference_only': True, 'production_ledger_access_required': False})
        else:
            portable_operators(original, target)
    agh = out / 'agh'
    agh.mkdir()
    if baseline_source:
        destination = agh / 'baseline-g000'
        destination.mkdir()
        safe_copy(baseline_source.parent, baseline_source.name, destination)
        (destination / baseline_source.name).rename(destination / 'source.tar')
        write_json(destination / 'provenance.json', {
            'role': 'original_H0_comparison_source', 'source_commit': AGH_BASELINE_COMMIT,
            'archive_sha256': AGH_BASELINE_ARCHIVE_SHA, 'tracked_source_files': 3625,
            'original_source_bytes_unmodified': True, 'incumbent_pointer_changed': False,
            'actual_Node_rebuild_from_this_archive_proven': False})
    shutil.copyfile(args.agh_archive, agh / 'source.tar.gz')
    if sha(agh / 'source.tar.gz') != AGH_ARCHIVE_SHA:
        raise ValueError('AGH archive copy changed')
    with tarfile.open(args.agh_archive, 'r:gz') as bundle:
        for name in ['LICENSE', 'NOTICE']:
            (agh / name).write_bytes(bundle.extractfile(name).read())
    for commit in ['1dde234563fbe19a30079c39411ae59ee9aff8d6', '1c11d9bf21421175668e47cd914cac7754bdb873', AGH_COMMIT]:
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', commit])
        inspect_bytes(patch, commit)
        (agh / (commit[:7] + '.patch')).write_bytes(patch)
    if candidate:
        destination = agh / 'candidate-g004'
        destination.mkdir()
        safe_copy(candidate.parent, candidate.name, destination)
        (destination / candidate.name).rename(destination / 'source.tar')
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', AGH_CANDIDATE_COMMIT])
        inspect_bytes(patch, AGH_CANDIDATE_COMMIT)
        (destination / 'source.patch').write_bytes(patch)
        write_json(destination / 'provenance.json', {'source_commit': AGH_CANDIDATE_COMMIT,
            'parent_source_commit': AGH_COMMIT, 'archive_sha256': AGH_CANDIDATE_ARCHIVE_SHA,
            'author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False,
            'scope': 'New recovery-hint source, separate from the original eight-arm G003 evidence.'})
    if args.demo:
        for source in sorted(args.demo.rglob('*')):
            if source.is_file():
                safe_copy(args.demo, source.relative_to(args.demo).as_posix(), out / 'demo')
    if next_source:
        destination = agh / 'candidate-g005'
        destination.mkdir()
        safe_copy(next_source.parent, next_source.name, destination)
        (destination / next_source.name).rename(destination / 'source.tar')
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', AGH_NEXT_COMMIT])
        inspect_bytes(patch, AGH_NEXT_COMMIT)
        (destination / 'source.patch').write_bytes(patch)
        write_json(destination / 'provenance.json', {'source_commit': AGH_NEXT_COMMIT,
            'parent_source_commit': AGH_CANDIDATE_COMMIT, 'archive_sha256': AGH_NEXT_ARCHIVE_SHA,
            'author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False,
            'scope': 'New source-bound terminal failure and approval recovery guidance; separate fresh V15 comparison.'})
    if latest_source:
        destination = agh / 'candidate-g006'
        destination.mkdir()
        safe_copy(latest_source.parent, latest_source.name, destination)
        (destination / latest_source.name).rename(destination / 'source.tar')
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', AGH_LATEST_COMMIT])
        inspect_bytes(patch, AGH_LATEST_COMMIT)
        (destination / 'source.patch').write_bytes(patch)
        write_json(destination / 'provenance.json', {'source_commit': AGH_LATEST_COMMIT,
            'parent_source_commit': AGH_NEXT_COMMIT, 'archive_sha256': AGH_LATEST_ARCHIVE_SHA,
            'author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False,
            'scope': 'Context reserve repair; fresh V16 performance comparison must be independently accepted.'})
    if structured_source:
        destination = agh / 'candidate-g007'
        destination.mkdir()
        safe_copy(structured_source.parent, structured_source.name, destination)
        (destination / structured_source.name).rename(destination / 'source.tar')
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', AGH_STRUCTURED_COMMIT])
        inspect_bytes(patch, AGH_STRUCTURED_COMMIT)
        (destination / 'source.patch').write_bytes(patch)
        write_json(destination / 'provenance.json', {'source_commit': AGH_STRUCTURED_COMMIT,
            'parent_source_commit': AGH_LATEST_COMMIT, 'archive_sha256': AGH_STRUCTURED_ARCHIVE_SHA,
            'author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False,
            'scope': 'Structured JSON model-view compaction; independent V17 controls, no search benefit claimed.'})
    if inline_source:
        destination = agh / 'candidate-g008'
        destination.mkdir()
        safe_copy(inline_source.parent, inline_source.name, destination)
        (destination / inline_source.name).rename(destination / 'source.tar')
        patch = subprocess.check_output(['git', '-C', str(args.agh_repo), 'show', '--format=fuller', '--binary', AGH_INLINE_COMMIT])
        inspect_bytes(patch, AGH_INLINE_COMMIT)
        (destination / 'source.patch').write_bytes(patch)
        write_json(destination / 'provenance.json', {'source_commit': AGH_INLINE_COMMIT,
            'parent_source_commit': AGH_STRUCTURED_COMMIT, 'archive_sha256': AGH_INLINE_ARCHIVE_SHA,
            'author': 'Codex outer loop', 'model_authorship_verified': False,
            'formal_source_performance_promoted': False,
            'scope': 'Use included terminal exception diagnostics; 78 actual localhost SDK bodies, new V19 performance not established.'})
    for source, generation in ((request_source, 'g009'), (correctness_source, 'g010'), (compiler_source, 'g011'), (unpromoted_source, 'g012'), (pending_source, 'g013'), (running_pending_source, 'g014'), (source_copy_source, 'g017')):
        if source:
            package_current_source(source, generation, agh)
    for source, name in [(args.ppt, 'presentation.pptx'), (args.video, 'demo.mp4')]:
        if source:
            safe_copy(source.parent, source.name, out / 'materials')
            (out / 'materials' / source.name).rename(out / 'materials' / name)
    (out / 'reproduce.py').write_text(REPRODUCE)
    (out / 'README.md').write_text(README)
    if candidate:
        with (out / 'README.md').open('a') as stream:
            stream.write('\nG004源码位于 `agh/candidate-g004/`，工程父版本为G003。完整V14的原H0/G004八臂已闭账，未通过原性能晋级门；此前G000/G003系列结果另行保留。\n')
    if next_source:
        with (out / 'README.md').open('a') as stream:
            stream.write('\nG005源码、patch和谱系位于 `agh/candidate-g005/`，工程父版本为G004。189项回归、551项guard及39次新运行包localhost请求通过；完整V15八臂未通过原性能晋级门。真实GPT-2保留长形状数值失败，未证明全形状强基线胜出。\n')
    if latest_source:
        with (out / 'README.md').open('a') as stream:
            stream.write('\nG006源码、patch和谱系位于 `agh/candidate-g006/`，工程父版本为G005。32K窗口压缩预留由16K改为8K；221相关回归、551guard与独立175运行包/39localhost及实际大输入资格通过；CPU工程修复不等同于V16搜索性能晋级。\n')
    if structured_source:
        with (out / 'README.md').open('a') as stream:
            stream.write('\nG007源码、patch和谱系位于 `agh/candidate-g007/`：只压缩匹配的完整工具JSON模型视图，原始事件/数字/字符串/权限封套保持；364相关回归、551guard及175文件新运行包/39次实际SDK资格通过。V18全部八臂160位置已闭账，原搜索收益及成本晋级门未通过。旧K02的GPU数值诊断保留四形状：局部48/48通过、参考接线4/4逐位相等、全模型仅1/4通过；该诊断未测性能。新的Welford实测结果单独保留。\n')
    with (out / 'README.md').open('a') as stream:
        stream.write('\n新GEMM应用入口 `--application packed-variant`，原H0版本在GPU0/3完整独立复测：普通FP16库2.08–2.45倍，强库边界更慢。精度诊断入口 `--application pretrained-precision --model-dir /your/local/gpt2`；仅短形状通过，长形状仍失败。原两次900秒应用失败与成本保留，两个入口默认仅CPU计划。\n')
        stream.write('\nWelford四形状原完整预训练输出逐位一致，48局部检查通过；原30轮配对有240GPU+240host样本，相对普通FP16库约0.90–0.95倍，没有端到端加速。另一个共同编译配置仅1/4形状通过原数值门，未计时。原失败、源码及全部公开报告均保留；数值复现入口仍只验证原四形状，不冒称性能复现。\n')
        if inline_source:
            stream.write('\nG008完整3628文件源码、patch和LICENSE/NOTICE位于 `agh/candidate-g008/`。368相关测试、551guard加2原skip、175文件新运行包和78实际本地SDK请求有证据；工程资格与搜索性能晋级分别验收。12算子原来源保持，CSR历史大比值针对原Python分段基线，不能替代强库或整模型比较。\n')
        if request_source:
            stream.write('\n当前已闭账证据：G008/V19、G009/V21的完整八臂均未通过原晋级门，H0仍为incumbent；G009完整源码及原patch位于 `agh/candidate-g009/`。V21的160位置、一个未获得候选的位置、失败和unknown预留都保留在原结果中。\n')
            stream.write('\n真实图计算入口：`python reproduce.py --application citation-graph --archive /your/local/arxiv.zip --out /tmp/new-graph-plan`。默认仅CPU计划，全量/2048节点窗口/边界三个案例；原83058288字节archive的SHA256须为49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276。数据不包含在ZIP、不自动下载。仅在自己的干净CUDA设备上显式加`--execute --gpu 0`才执行原900秒、30配对轮、五种强库与原数值门。未训练的固定随机投影不证明分类精度。原两设备全量强库收益约1.5905/1.5920倍，1080 GPU和1080主机原样本及实际浏览器展示资格包含在包内；新独立GPU复现仍待验收。原CSR跨设备复测约3.358倍对编译库与约730.74倍对Python基线，基线范围不同。\n')
        if correctness_source:
            stream.write('\nG010完整源码、LICENSE/NOTICE及原patch位于 `agh/candidate-g010/`，工程父版本为G009。实际39个本地SDK请求验证结构化数值失败提示可达HTTP正文；源码作者为外层Codex。V22一个DEV消融对从8/20正确变为12/20，但全20几何均值比0.9733、95%CI[0.8481,1.0953]、GPU成本比1.1367，未过原门。独立H0/G010的V23全部八组160位置已闭账、160评测、0ENV；DEV正确项[10,11,9]对[8,5,9]，综合比1.01650、95%CI[0.92580,1.13116]，前两组GPU成本门未过。保留集两版均10/20正确、综合比1.02779，不能补救开发门失败；H0仍incumbent，RSI未证明。\n')
        if compiler_source:
            stream.write('\nG011完整3628文件源码、LICENSE/NOTICE及原patch位于 `agh/candidate-g011/`，工程父版本为G010。终态编译失败的静态修复提示有独立CPU构建和39次实际本地SDK资格，原guard两个跳过保留。V24两组已闭账：G011正确10/20对G010的11/20，缺少一项最终评测，综合比1.00119、95%CI[0.89885,1.11702]，未通过原门。失败、未知成本与拒绝保持；H0仍incumbent，RSI未证明。\n')
        if unpromoted_source:
            stream.write('\nG012完整3628文件源码、LICENSE/NOTICE及原patch位于 `agh/candidate-g012/`，直接从G010分支，不叠加G011。437项相关测试、完整构建和39次本地SDK资格通过，原guard两个跳过保留。V25两组40位置已完整评测：正确10/20对父版本13/20，全20综合比1.00918，95%CI[0.92567,1.10417]；token/USD/GPU成本比1.18873/1.23238/1.30298，未通过原门。完整历史31576请求及未知预留保留；无晋级或RSI。源码作者为外层Codex。\n')
        if pending_source:
            stream.write('\nG013完整3629文件源码、LICENSE/NOTICE及原patch位于 `agh/candidate-g013/`，直接从G010分支。首次待完成观测留档后，在原剩余超时内最多续查一次；取消、权限、24请求和四候选上限保留。525项相关测试、175文件独立运行包及48次实际本地SDK正文验证通过，原guard两项跳过保留。合成作业流程模型请求3→2、CLI查询均为2次，只证明执行行为；V26原两组闭账均正确11/20，G013最终评测19/20缺失K16，综合比1.02375、置信区间含1、GPU成本比1.12764，原门未过，无晋级或RSI。作者为外层Codex。\n')
        if running_pending_source:
            stream.write('\nG014完整源码、LICENSE/NOTICE与patch位于 `agh/candidate-g014/`，工程父版G013。仅对运行中的作业继续有界续查；排队结果交还模型。526项相关回归、175运行文件、52次实际SDK正文资格通过；运行路径两版模型请求均为2次，排队路径G014为3次、G013为2次，CLI均查询两次。保留真实取舍；付费对照待测，无晋级或RSI，作者为外层Codex。\n')
        if source_copy_source:
            stream.write('\nG017完整3630文件源码、原LICENSE/NOTICE和patch位于 `agh/candidate-g017/`，工程父版为exact G014；未继承未接受的G016。成功write后仅对同lane、有可信因果回执的重复实现追加提示，实际写入与hook保持；不禁止正确候选的有意复测。独立546项相关检查、175运行文件、67个实际本地SDK HTTP体通过；复制流程父子均5请求、4次实际write。V30四卡20任务消融仍待GPU准入和付费实测，不宣称收益或RSI，源码作者为外层Codex。V29的G014/G016对照已关闭：正确10/12，最终20/19，K16缺失；综合比0.978851、置信区间含1、GPU成本比1.444701，不晋级。完整历史33912请求和24482未知预留保留。\n')
    (out / 'requirements-reproduction.txt').write_text(
        '# Recorded measured environment, Python 3.11.10, CUDA runtime 12.1\n'
        'torch==2.5.1+cu121\ntriton==3.1.0\nnumpy==1.26.4\n'
        'transformers==4.46.1\ntokenizers==0.20.2\nsafetensors==0.4.5\nscipy==1.14.1\n')
    (out / 'THIRD_PARTY_NOTICES.md').write_text(
        '# Third-party source and runtime dependencies\n\n'
        'The AGH snapshot includes its Apache-2.0 LICENSE and NOTICE, including third-party MIT notices. '
        'All tracked vendored notices remain inside the original source archive.\n\n'
        'Measured runtime: Python3.11.10 (PSF), PyTorch2.5.1+cu121 (BSD-3-Clause), '
        'Triton3.1.0 (MIT), NumPy1.26.4 and SciPy1.14.1 (BSD-3-Clause). These runtime packages are not redistributed in this source bundle. '
        'Pretrained application uses Transformers4.46.1, Tokenizers0.20.2 and Safetensors0.4.5 (Apache-2.0); '
        'GPT-2 original model weights remain a local external reference, not redistributed in the ZIP. '
        'Install them from their official distributions and retain their licenses.\n\n'
        'Optional delivery tools: python-pptx1.0.2 (MIT), Playwright1.63.0 (Apache-2.0), '
        'imageio-ffmpeg0.6.0 (BSD-2-Clause), Pillow (HPND), Noto CJK fonts (SIL OFL). '
        'Tool executables and fonts are not included.\n')
    write_json(out / 'experiment-template.json', {'model': 'mimo-v2.6-pro', 'thinking': 'off',
        'model_slots': ['primary', 'compaction', 'fast', 'escalation', 'verifier'], 'context_window': 32768,
        'output_tokens': 4096, 'task_concurrency': 20, 'evaluator_concurrency': 3, 'max_workers': 10,
        'per_task_native_requests': 24, 'per_task_candidates': 4, 'per_task_seconds': 1200,
        'global_token_limit': None, 'global_USD_limit': None, 'global_GPU_seconds_limit': None,
        'global_deadline': None, 'no_champion_inheritance': True, 'credential_configuration': 'TEAM_PRIVATE_ENVIRONMENT',
        'source_commit': AGH_COMMIT, 'formal_source_performance_promoted': False})
    if candidate:
        template = json.loads((out / 'experiment-template.json').read_bytes())
        template.update(source_commit=AGH_CANDIDATE_COMMIT, parent_source_commit='54b05f2a959b5f6328ac2bf572fbd981bf567094',
                        evaluator_concurrency=4, common_clean_gpu_pool=[0, 1, 3, 4],
                        comparison='Original H0 versus G004; source edits only after all predeclared arms close.')
        write_json(out / 'experiment-template-g004.json', template)
    if next_source:
        template = json.loads((out / 'experiment-template-g004.json').read_bytes())
        template.update(source_commit=AGH_NEXT_COMMIT, engineering_parent_source_commit=AGH_CANDIDATE_COMMIT,
                        comparison='Original H0 versus G005; fresh DEV071/072/073 and independent holdout075.')
        write_json(out / 'experiment-template-g005.json', template)
    if latest_source:
        template = json.loads((out / 'experiment-template-g005.json').read_bytes())
        template.update(source_commit=AGH_LATEST_COMMIT, engineering_parent_source_commit=AGH_NEXT_COMMIT,
                        comparison='Original H0 versus G006; fresh DEV081/082/083 and independent holdout085.')
        write_json(out / 'experiment-template-g006.json', template)
    if structured_source:
        template = json.loads((out / 'experiment-template-g006.json').read_bytes())
        template.update(source_commit=AGH_STRUCTURED_COMMIT, engineering_parent_source_commit=AGH_LATEST_COMMIT,
                        comparison='Original H0 versus G007; fresh DEV091/092/093 and independent holdout095.')
        write_json(out / 'experiment-template-g007.json', template)
    if inline_source:
        template = json.loads((out / 'experiment-template-g007.json').read_bytes())
        template.update(source_commit=AGH_INLINE_COMMIT, engineering_parent_source_commit=AGH_STRUCTURED_COMMIT,
                        evaluator_concurrency=2, common_clean_gpu_pool=[3, 4],
                        comparison='Original H0 versus G008; three fresh DEV pairs and independent holdout, no champion inheritance.')
        write_json(out / 'experiment-template-g008.json', template)
    if request_source:
        template = json.loads((out / 'experiment-template-g008.json').read_bytes())
        template = current_source_template(template, 'g009',
                        'Original H0 versus G009; closed V21 failed unchanged promotion gates.',
                        evaluator_concurrency=3, common_clean_gpu_pool=[1, 3, 4],
                        RSI_proven=False)
        write_json(out / 'experiment-template-g009.json', template)
    if correctness_source:
        template = json.loads((out / 'experiment-template-g009.json').read_bytes())
        template = current_source_template(template, 'g010',
                        'Original H0 versus G010; all eight V23 arms closed, original mandatory promotion gates failed.')
        write_json(out / 'experiment-template-g010.json', template)
    if compiler_source:
        template = json.loads((out / 'experiment-template-g010.json').read_bytes())
        template = current_source_template(template, 'g011',
                        'G010 versus G011; closed V24 failed original correctness and performance gates.',
                        ablation_only=True, promotion_permitted=False)
        write_json(out / 'experiment-template-g011.json', template)
    if unpromoted_source:
        template = json.loads((out / 'experiment-template-g010.json').read_bytes())
        template = current_source_template(template, 'g012',
                        'G010 versus G012; closed V25 failed original correctness, confidence and cost gates.',
                        ablation_only=True, promotion_permitted=False, RSI_proven=False)
        write_json(out / 'experiment-template-g012.json', template)
    if pending_source:
        template = json.loads((out / 'experiment-template-g010.json').read_bytes())
        template = current_source_template(template, 'g013',
                        'G010 versus G013, closed DEV202610261 original V26 ablation; missing K16, confidence and GPU cost gates failed, no promotion.',
                        ablation_only=True, promotion_permitted=False)
        write_json(out / 'experiment-template-g013.json', template)
    if running_pending_source:
        template = json.loads((out / 'experiment-template-g013.json').read_bytes())
        template = current_source_template(template, 'g014',
                        'G013 versus G014 running-only continuation; proposed matched four-device pair, registration and paid performance pending.',
                        evaluator_concurrency=4, common_clean_gpu_pool=[0, 1, 3, 4],
                        ablation_only=True, promotion_permitted=False, measurement_registration_frozen=False)
        write_json(out / 'experiment-template-g014.json', template)
    if source_copy_source:
        template = json.loads((out / 'experiment-template-g014.json').read_bytes())
        template = current_source_template(template, 'g017',
            'Exact G014 versus G017, fresh V30 DEV ablation; original GPU GO and paid source performance pending.',
            evaluator_concurrency=4, common_clean_gpu_pool=[0, 1, 3, 4],
            ablation_only=True, promotion_permitted=False, measurement_registration_frozen=True,
            clean_GPU_GO_passed=False, paid_source_performance_proven=False,
            actual_CPU_SDK_HTTP_bodies=67)
        write_json(out / 'experiment-template-g017.json', template)
    result = write_inventory(out)
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise ValueError('delivery ZIP must be fresh')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in sorted(out.rglob('*')):
            if source.is_file():
                bundle.write(source, source.relative_to(out).as_posix())
    if archive.stat().st_size > 200_000_000:
        raise ValueError('delivery ZIP exceeds the original 200 MB delivery ceiling')
    receipt = {**result, 'status': 'ASSEMBLED_HASH_VERIFIED_PREVIEW', 'inventory_status': result['status'], 'bundle': str(out),
        'zip_sha256': sha(archive), 'zip_bytes': archive.stat().st_size,
        'agh_source_commit': AGH_COMMIT, 'agh_source_archive_sha256': AGH_ARCHIVE_SHA,
        'CPU_reproduction': 'PENDING_ACTUAL_ASSEMBLED_RUN', 'published': False, 'submitted': False}
    write_json(out.parent / (out.name + '-build-receipt.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--agh-archive', type=Path, required=True)
    parser.add_argument('--agh-baseline-source', type=Path)
    parser.add_argument('--agh-repo', type=Path, required=True)
    parser.add_argument('--agh-candidate-source', type=Path)
    parser.add_argument('--agh-next-source', type=Path)
    parser.add_argument('--agh-latest-source', type=Path)
    parser.add_argument('--agh-structured-source', type=Path)
    parser.add_argument('--agh-inline-source', type=Path)
    parser.add_argument('--agh-request-budget-source', type=Path)
    parser.add_argument('--agh-correctness-source', type=Path)
    parser.add_argument('--agh-compiler-source', type=Path)
    parser.add_argument('--agh-unpromoted-source', type=Path)
    parser.add_argument('--agh-pending-source', type=Path)
    parser.add_argument('--agh-running-pending-source', type=Path)
    parser.add_argument('--agh-source-copy-source', type=Path)
    parser.add_argument('--demo', type=Path)
    parser.add_argument('--ppt', type=Path)
    parser.add_argument('--video', type=Path)
    print(json.dumps(build(parser.parse_args())))


if __name__ == '__main__':
    main()
