#!/bin/bash
sudo env "PATH=$(pwd)/venv/bin:$PATH" $(pwd)/venv/bin/python $(pwd)/venv/bin/afusion run

