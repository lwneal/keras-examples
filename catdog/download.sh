#!/bin/bash
which kg || echo "Install kg with: pip install kaggle-cli"
kg download -c dogs-vs-cats-redux-kernels-edition
