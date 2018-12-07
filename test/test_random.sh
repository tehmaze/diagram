#!/bin/bash

set -o pipefail -o errexit -o nounset -o xtrace

cd $(dirname "${BASH_SOURCE[0]}")/..

./bin/random 100 | python diagram.py
./bin/random 100 | python diagram.py -H
./bin/random 100 | python diagram.py -V
./bin/random 100 | python diagram.py -b
./bin/random 100 | python diagram.py -bH
./bin/random 100 | python diagram.py -bV

echo -e '1\n2\n3\n2' | diagram -H > output
# We check that the number of lines in the output is [input size] + 1
[ $(wc -l output | cut -d' ' -f1) -eq 5 ] || exit 1
