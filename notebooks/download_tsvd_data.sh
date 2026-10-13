#!/usr/bin/env bash
# author: laquitainesteeve@gmail.com
# execution time: 15 secs

# Create target directory if it doesn't exist
mkdir -p data/tvsd
 
# download THINGS_normMUA.mat from GIN to data/tvsd/ (5 secs)
URL="https://gin.g-node.org/doi/TVSD/raw/master/monkeyN/THINGS_normMUA.mat"
OUTPUT="data/tvsd/monkeyN_THINGS_normMUA.mat"
curl -fL "$URL" -o "$OUTPUT"      # -f: fail on an HTTP error instead of saving the error page

# download things_imgs.mat from GIN to data/tvsd/ (32 MB)
# (the URL must contain /raw/: /src/ returns the web page of the file, not the file)
URL="https://gin.g-node.org/doi/TVSD/raw/master/monkeyN/_logs/things_imgs.mat"
OUTPUT="data/tvsd/things_imgs.mat"
curl -fL "$URL" -o "$OUTPUT"      # -f: fail on an HTTP error instead of saving the error page