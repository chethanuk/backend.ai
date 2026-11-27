from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from yarl import URL

from ai.backend.client.config import APIConfig, set_config
from ai.backend.client.exceptions import BackendConnectError
from ai.backend.client.func.vfolder import VFolderByName
from ai.backend.client.session import api_session

_MAX_DOWNLOAD_RETRIES = 3


@dataclass(frozen=True)
class _FailingDownloadCase:
    id: str
    connection_error: aiohttp.ClientConnectorError
    expected_error: type[Exception]
    expected_attempts: int


@dataclass(frozen=True)
class _FailingDownload:
    storage_proxy_get: mock.MagicMock
    vfolder: VFolderByName
    file_path: Path


class TestVFolderDownload:
    @pytest.fixture
    def failing_download(
        self, case: _FailingDownloadCase, tmp_path: Path
    ) -> Iterator[_FailingDownload]:
        with mock.patch.object(
            aiohttp.ClientSession, "get", side_effect=case.connection_error
        ) as storage_proxy_get:
            yield _FailingDownload(
                storage_proxy_get=storage_proxy_get,
                vfolder=VFolderByName("myfolder"),
                file_path=tmp_path / "data.bin",
            )

    @pytest.mark.parametrize(
        "case",
        [
            _FailingDownloadCase(
                id="certificate-not-verified",
                connection_error=aiohttp.ClientConnectorCertificateError(
                    mock.MagicMock(host="storage.example", port=6022),
                    ssl.SSLCertVerificationError(1, "certificate verify failed"),
                ),
                expected_error=BackendConnectError,
                expected_attempts=1,
            ),
            _FailingDownloadCase(
                id="connection-refused",
                connection_error=aiohttp.ClientConnectorError(
                    mock.MagicMock(host="storage.example", port=6022),
                    ConnectionRefusedError(111, "Connection refused"),
                ),
                expected_error=RuntimeError,
                expected_attempts=_MAX_DOWNLOAD_RETRIES,
            ),
        ],
        ids=lambda case: case.id,
    )
    async def test_retries_only_errors_other_than_certificate_verification(
        self, case: _FailingDownloadCase, failing_download: _FailingDownload
    ) -> None:
        with pytest.raises(case.expected_error):
            await failing_download.vfolder._download_file(
                failing_download.file_path,
                URL("https://storage.example:6022/download"),
                chunk_size=1024,
                max_retries=_MAX_DOWNLOAD_RETRIES,
                show_progress=False,
            )

        assert failing_download.storage_proxy_get.call_count == case.expected_attempts


class TestVFolderDownloadReadTimeout:
    @pytest.fixture
    async def stalled_storage_proxy(self) -> AsyncIterator[TestServer]:
        release = asyncio.Event()

        async def stalled_handler(request: web.Request) -> web.StreamResponse:
            await release.wait()
            return web.Response(body=b"never sent")

        app = web.Application()
        app.router.add_get("/download", stalled_handler)
        async with TestServer(app) as server:
            try:
                yield server
            finally:
                release.set()

    @pytest.mark.parametrize("config_source", ["global_config", "api_session"])
    async def test_stalled_storage_proxy_honors_configured_read_timeout(
        self,
        tmp_path: Path,
        defconfig: APIConfig,
        stalled_storage_proxy: TestServer,
        config_source: str,
    ) -> None:
        config = APIConfig(
            endpoint=defconfig.endpoint,
            access_key=defconfig.access_key,
            secret_key=defconfig.secret_key,
            read_timeout=0.2,
        )
        session_token = None
        if config_source == "api_session":
            session_token = api_session.set(mock.Mock(config=config))
        else:
            set_config(config)
        try:
            # The default aiohttp timeout (300s) would outlast this guard.
            with pytest.raises(aiohttp.ServerTimeoutError):
                await asyncio.wait_for(
                    VFolderByName("myfolder")._download_file(
                        tmp_path / "data.bin",
                        stalled_storage_proxy.make_url("/download"),
                        chunk_size=1024,
                        max_retries=1,
                        show_progress=False,
                    ),
                    timeout=5,
                )
        finally:
            if session_token is not None:
                api_session.reset(session_token)
            set_config(defconfig)
