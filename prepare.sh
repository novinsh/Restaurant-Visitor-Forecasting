#!/bin/bash

mkdir -p data # create data directory if not existing

echo 'downloading datasets:'
curl -L -o data/everything.zip https://www.dropbox.com/sh/3ruqzzufte7xj6z/AADM0U-hQMDMVAdrvYoOvk-Fa?dl=1

echo 'unzipping datasets..'
unzip data/everything.zip -d data

echo 'clean up..'
rm -vfd data/everything.zip
echo "I'm done! your turn, let the awesome begin.."


