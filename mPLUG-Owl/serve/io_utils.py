# -*- coding: utf-8 -*- 
"""
@Time : 2023-03-23 11:42 
@Author : zhimiao.chh 
@Desc : 
"""

import re
import os
import sys
import shutil
import hashlib
from io import StringIO, BytesIO
from contextlib import contextmanager
from typing import List
from datetime import datetime, timedelta


class IO:
    @staticmethod
    def register(options):
        pass

    def open(self, path: str, mode: str):
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def move(self, src: str, dst: str):
        raise NotImplementedError

    def copy(self, src: str, dst: str):
        raise NotImplementedError

    def makedirs(self, path: str, exist_ok=True):
        raise NotImplementedError

    def remove(self, path: str):
        raise NotImplementedError

    def listdir(self, path: str, recursive=False, full_path=False, contains=None):
        raise NotImplementedError

    def isdir(self, path: str) -> bool:
        raise NotImplementedError

    def isfile(self, path: str) -> bool:
        raise NotImplementedError

    def abspath(self, path: str) -> str:
        raise NotImplementedError

    def last_modified(self, path: str) -> datetime:
        raise NotImplementedError

    def md5(self, path: str) -> str:
        hash_md5 = hashlib.md5()
        with self.open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    re_remote = re.compile(r'(oss|https?)://')

    def islocal(self, path: str) -> bool:
        return not self.re_remote.match(path.lstrip())


class DefaultIO(IO):
    __name__ = 'DefaultIO'

    def _check_path(self, path):
        if not self.islocal(path):
            raise RuntimeError(
                'Credentials must be provided to use oss path. '
                'Make sure you have created "user/modules/oss_credentials.py" according to ReadMe.')

    def open(self, path, mode='r'):
        self._check_path(path)
        path = self.abspath(path)
        return open(path, mode=mode)

    def exists(self, path):
        self._check_path(path)
        path = self.abspath(path)
        return os.path.exists(path)

    def move(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        shutil.move(src, dst)

    def copy(self, src, dst):
        self._check_path(src)
        self._check_path(dst)
        src = self.abspath(src)
        dst = self.abspath(dst)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass

    def makedirs(self, path, exist_ok=True):
        self._check_path(path)
        path = self.abspath(path)
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path):
        self._check_path(path)
        path = self.abspath(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    def listdir(self, path, recursive=False, full_path=False, contains=None):
        self._check_path(path)
        path = self.abspath(path)
        contains = contains or ''
        if recursive:
            files = (os.path.join(dp, f) if full_path else f for dp, dn, fn in os.walk(path) for f in fn)
            files = [file for file in files if contains in file]
        else:
            files = os.listdir(path)
            if full_path:
                files = [os.path.join(path, file) for file in files if contains in file]
        return files

    def isdir(self, path):
        return os.path.isdir(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def abspath(self, path):
        return os.path.abspath(path)

    def last_modified(self, path):
        return datetime.fromtimestamp(os.path.getmtime(path))


class OSS(DefaultIO):
    "Mixed IO module to support both system-level and OSS IO methods"
    __name__ = 'OSS'

    def __init__(self, access_key_id: str, access_key_secret: str, region_bucket: List[List[str]]):
        """
        the value of "region_bucket" should be something like [["cn-hangzhou", "<yourBucketName>"], ["cn-zhangjiakou", "<yourBucketName>"]],
        specifying your buckets and corresponding regions
        """
        from oss2 import Auth, Bucket, ObjectIterator
        super().__init__()
        self.ObjectIterator = ObjectIterator
        self.auth = Auth(access_key_id, access_key_secret)
        self.buckets = {
            bucket_name: Bucket(self.auth, f'http://oss-{region}.aliyuncs.com', bucket_name)
            for region, bucket_name in region_bucket
        }
        self.oss_pattern = re.compile(r'oss://([^/]+)/(.+)')

    def _split_name(self, path):
        m = self.oss_pattern.match(path)
        if not m:
            raise IOError(f'invalid oss path: "{path}", should be "oss://<bucket_name>/path"')
        bucket_name, path = m.groups()
        return bucket_name, path

    def _split(self, path):
        bucket_name, path = self._split_name(path)
        try:
            bucket = self.buckets[bucket_name]
        except KeyError:
            raise IOError(f'Bucket {bucket_name} not registered in oss_credentials.py')
        return bucket, path

    def open(self, full_path, mode='r'):
        if not full_path.startswith('oss://'):
            return super().open(full_path, mode)

        bucket, path = self._split(full_path)
        with mute_stderr():
            path_exists = bucket.object_exists(path)
        if 'w' in mode:
            if path_exists:
                bucket.delete_object(path)
            if 'b' in mode:
                return BinaryOSSFile(bucket, path)
            return OSSFile(bucket, path)
        elif mode == 'a':
            position = bucket.head_object(path).content_length if path_exists else 0
            return OSSFile(bucket, path, position=position)
        else:
            if not path_exists:
                raise FileNotFoundError(full_path)
            obj = bucket.get_object(path)
            # auto cache large files to avoid memory issues
            # if obj.content_length > 30 * 1024 ** 2:  # 30M
            #     from da.utils import cache_file
            #     path = cache_file(full_path)
            #     return super().open(path, mode)
            if mode == 'rb':
                # TODO for a large file, this will load the whole file into memory
                return NullContextWrapper(BytesIO(obj.read()))
            else:
                assert mode == 'r'
                return NullContextWrapper(StringIO(obj.read().decode()))

    def exists(self, path):
        if not path.startswith('oss://'):
            return super().exists(path)

        bucket, _path = self._split(path)
        # if file exists
        exists = self._file_exists(bucket, _path)
        # if directory exists
        if not exists:
            try:
                self.listdir(path)
                exists = True
            except FileNotFoundError:
                pass
        return exists

    def _file_exists(self, bucket, path):
        with mute_stderr():
            return bucket.object_exists(path)

    def move(self, src, dst):
        if not src.startswith('oss://') and not dst.startswith('oss://'):
            return super().move(src, dst)
        self.copy(src, dst)
        self.remove(src)

    def copy(self, src, dst):
        cloud_src = src.startswith('oss://')
        cloud_dst = dst.startswith('oss://')
        if not cloud_src and not cloud_dst:
            return super().copy(src, dst)

        # download
        if cloud_src and not cloud_dst:
            bucket, src = self._split(src)
            obj = bucket.get_object(src)
            if obj.content_length > 100 * 1024 ** 2:  # 100M
                from tqdm import tqdm
                progress = None

                def callback(i, n):
                    nonlocal progress
                    if progress is None:
                        progress = tqdm(total=n, unit='B', unit_scale=True, unit_divisor=1024, leave=False,
                                        desc=f'downloading')
                    progress.update(i - progress.n)

                bucket.get_object_to_file(src, dst, progress_callback=callback)
                if progress is not None:
                    progress.close()
            else:
                bucket.get_object_to_file(src, dst)
            return
        bucket, dst = self._split(dst)
        # upload
        if cloud_dst and not cloud_src:
            bucket.put_object_from_file(dst, src)
            return
        # copy between oss paths
        if src != dst:
            src_bucket_name, src = self._split_name(src)
            bucket.copy_object(src_bucket_name, src, dst)
        # TODO: support large file copy
        # https://help.aliyun.com/document_detail/88465.html?spm=a2c4g.11174283.6.882.4d157da2mgp3xc

    def listdir(self, path, recursive=False, full_path=False, contains=None):
        if not path.startswith('oss://'):
            return super().listdir(path, recursive, full_path, contains)

        bucket, path = self._split(path)
        path = path.rstrip('/') + '/'
        files = [obj.key for obj in self.ObjectIterator(bucket, prefix=path, delimiter='' if recursive else '/')]
        try:
            files.remove(path)
        except ValueError:
            pass
        if full_path:
            files = [f'oss://{bucket.bucket_name}/{file}' for file in files]
        else:
            files = [file[len(path):] for file in files]
        if not files:
            raise FileNotFoundError(f'No such directory: oss://{bucket.bucket_name}/{path}')
        files = [file for file in files if (contains or '') in file]
        return files

    def remove(self, path):
        if not path.startswith('oss://'):
            return super().remove(path)

        if self.isfile(path):
            paths = [path]
        else:
            paths = self.listdir(path, recursive=True, full_path=True)
        for path in paths:
            bucket, path = self._split(path)
            bucket.delete_object(path)

    def makedirs(self, path, exist_ok=True):
        # there is no need to create directory in oss
        if not path.startswith('oss://'):
            return super().makedirs(path)

    def isdir(self, path):
        if not path.startswith('oss://'):
            return super().isdir(path)
        return self.exists(path.rstrip('/') + '/')

    def isfile(self, path):
        if not path.startswith('oss://'):
            return super().isdir(path)
        return self.exists(path) and not self.isdir(path)

    def abspath(self, path):
        if not path.startswith('oss://'):
            return super().abspath(path)
        return path

    def authorize(self, path):
        if not path.startswith('oss://'):
            raise ValueError('Only oss path can use "authorize"')
        import oss2
        bucket, path = self._split(path)
        bucket.put_object_acl(path, oss2.OBJECT_ACL_PUBLIC_READ)

    def last_modified(self, path):
        if not path.startswith('oss://'):
            return super().last_modified(path)
        bucket, path = self._split(path)
        return datetime.strptime(
            bucket.get_object_meta(path).headers['Last-Modified'],
            r'%a, %d %b %Y %H:%M:%S %Z'
        ) + timedelta(hours=8)


class OSSFile:
    def __init__(self, bucket, path, position=0):
        self.position = position
        self.bucket = bucket
        self.path = path
        self.buffer = StringIO()

    def write(self, content):
        # without a "with" statement, the content is written immediately without buffer
        # when writing a large batch of contents at a time, this will be quite slow
        import oss2
        buffer = self.buffer.getvalue()
        if buffer:
            content = buffer + content
            self.buffer.close()
            self.buffer = StringIO()
        try:
            result = self.bucket.append_object(self.path, self.position, content)
            self.position = result.next_position
        except oss2.exceptions.PositionNotEqualToLength:
            raise RuntimeError(
                f'Race condition detected. It usually means multiple programs were writing to the same file'
                f'oss://{self.bucket.bucket_name}/{self.path} (Error 409: PositionNotEqualToLength)')
        except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
            self.buffer.write(content)
            sys.stderr.write(str(e) + f'when writing to oss://{self.bucket.bucket_name}/{self.path}. Content buffered.')

    def flush(self):
        "Dummy method for compatibility."
        pass

    def close(self):
        "Dummy method for compatibility."
        pass

    def seek(self, position):
        self.position = position

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        import oss2
        try:
            self.bucket.append_object(self.path, self.position, self.buffer.getvalue())
        except oss2.exceptions.RequestError as e:
            # TODO test whether this works
            if 'timeout' not in str(e):
                raise e
            # retry if timeout
            import time
            time.sleep(5)
            self.bucket.append_object(self.path, self.position, self.buffer.getvalue())


class BinaryOSSFile:
    def __init__(self, bucket, path):
        self.bucket = bucket
        self.path = path
        self.buffer = BytesIO()

    def __enter__(self):
        return self.buffer

    def __exit__(self, *args):
        self.bucket.put_object(self.path, self.buffer.getvalue())


class NullContextWrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __iter__(self):
        return self._obj.__iter__()

    def __next__(self):
        return self._obj.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@contextmanager
def ignore_io_error(msg=''):
    import oss2
    try:
        yield
    except (oss2.exceptions.RequestError, oss2.exceptions.ServerError) as e:
        sys.stderr.write(str(e) + ' ' + msg)


@contextmanager
def mute_stderr():
    cache = sys.stderr
    sys.stderr = StringIO()
    try:
        yield None
    finally:
        sys.stderr = cache