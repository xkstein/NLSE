#!/bin/bash

sudo py-spy record -r 10000 -f chrometrace -- python breather.py -noplot
#sudo py-spy record -r 10000 -o profile.svg -- python breather.py -noplot
