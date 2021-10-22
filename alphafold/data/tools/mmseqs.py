# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library to run MMSeqs from Python."""

from concurrent import futures
import glob
import os
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import request

from absl import logging
from pathlib import Path

from alphafold.data.tools import utils
# Internal import (7716).


def run_task(cmd: str, task_name: str):
    logging.info(f'Launching subprocess: {cmd}')
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    with utils.timing(task_name):
        _, stderr = process.communicate()
        retcode = process.wait()

    if retcode:
        raise RuntimeError(f'{task_name} failed: stderr: ' +
                           stderr.decode('utf-8'))


class MMSeqs:
    """Python wrapper of the MMSeqs binary."""
    def __init__(self,
                 *,
                 binary_path: str,
                 uniref50_database_path: str,
                 mgnify_database_path: str,
                 small_bfd_database_path: str,
                 tmp_dir: Path,
                 n_cpu: int = 32,
                 n_iter: int = 3,
                 e_value: float = 0.1):
        """Initializes the Python MMSeqs wrapper.

        Args:
          binary_path: The path to the mmseqs executable.
          uniref50_database_path: Path to uniref50 index
          mgnify_database_path: Path to mgnify index
          small_bfd_database_path: Path to small bfd index
          tmp_dir: Path,
          n_cpu: The number of CPUs to give MMSeqs.
          n_iter: The number of MMSeqs iterations.
          e_value: The E-value, see MMSeqs docs for more details.
        """
        self.binary_path = binary_path
        self.uniref50_database_path = uniref50_database_path
        self.mgnify_database_path = mgnify_database_path
        self.small_bfd_database_path = small_bfd_database_path
        self.tmp_dir = tmp_dir

        if not os.path.exists(self.uniref50_database_path):
            logging.error('Could not find MMSeqs database %s', self.uniref50_database_path)
            raise ValueError(f'Could not find MMSeqs database {self.uniref50_database_path}')

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value

    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        """Queries the database using MMSeqs."""

        with utils.tmpdir_manager(base_dir=self.tmp_dir) as query_tmp_dir:
            tmp_path: str = os.path.join(query_tmp_dir, 'tmp')
            input_fasta_dir: str = os.path.dirname(input_fasta_path)
            name: str = os.path.basename(input_fasta_path).split('.')[0]
            input_fasta_database: str = os.path.join(input_fasta_dir, name)

            if not os.path.exists(input_fasta_database):
                cmd = f"/usr/bin/time -v {self.binary_path} createdb {input_fasta_path} {input_fasta_database}"
                run_task(cmd, 'MMSeqs [ Creating query database ]')

            uniref_result_db: str = os.path.join(input_fasta_dir,
                                                 f'{name}_uniref')
            uniref_table: str = uniref_result_db + '.tab'
            if not os.path.exists(uniref_table):
                cmd = f"/usr/bin/time -v {self.binary_path} search {input_fasta_database} {self.uniref50_database_path} {uniref_result_db} {tmp_path} -e {self.e_value} --num-iterations 2 -s 4 -a"  # --start-sens 1 --sens-steps 3 -s 7 --num-iterations 3
                run_task(cmd, 'MMSeqs [ Searching query against uniref ]')
                cmd = f"/usr/bin/time -v {self.binary_path} convertalis {input_fasta_database} {self.uniref50_database_path} {uniref_result_db} {uniref_table} --format-output query,target,evalue,qaln,taln"
                run_task(
                    cmd,
                    'MMSeqs [ Converting mmseqs uniref result database to table ]'
                )

            mgnify_result_db: str = os.path.join(input_fasta_dir,
                                                 f'{name}_mgnify')
            mgnify_table: str = mgnify_result_db + '.tab'
            if not os.path.exists(mgnify_table):
                cmd = f"/usr/bin/time -v {self.binary_path} search {uniref_result_db} {self.mgnify_database_path} {mgnify_result_db} {tmp_path} -e {self.e_value} -s 4 -a"
                run_task(cmd, 'MMSeqs [ Searching query against mgnify ]')
                cmd = f"/usr/bin/time -v {self.binary_path} convertalis {uniref_result_db} {self.mgnify_database_path} {mgnify_result_db} {mgnify_table} --format-output query,target,evalue,qaln,taln"
                run_task(
                    cmd,
                    'MMSeqs [ Converting mmseqs mgnify result database to table ]'
                )

            small_bfd_result_db: str = os.path.join(input_fasta_dir,
                                                    f'{name}_small_bfd')
            small_bfd_table: str = small_bfd_result_db + '.tab'
            if not os.path.exists(small_bfd_table):
                cmd = f"/usr/bin/time -v {self.binary_path} search {uniref_result_db} {self.small_bfd_database_path} {small_bfd_result_db} {tmp_path} -e {self.e_value} -s 4 -a"
                run_task(cmd, 'MMSeqs [ Searching query against small_bfd ]')
                cmd = f"/usr/bin/time -v {self.binary_path} convertalis {uniref_result_db} {self.small_bfd_database_path} {small_bfd_result_db} {small_bfd_table} --format-output query,target,evalue,qaln,taln"
                run_task(
                    cmd,
                    'MMSeqs [ Converting mmseqs small_bfd result database to table ]'
                )

            with open(uniref_table, 'r') as F:
                uniref_result = F.read()
            with open(mgnify_table, 'r') as F:
                mgnify_result = F.read()
            with open(small_bfd_table, 'r') as F:
                small_bfd_result = F.read()
            all_results = '\n'.join(
                [uniref_result, mgnify_result, small_bfd_result])

        raw_output = dict(sto=sto,
                          tbl=tbl,
                          stderr=stderr,
                          n_iter=self.n_iter,
                          e_value=self.e_value)

        return raw_output
