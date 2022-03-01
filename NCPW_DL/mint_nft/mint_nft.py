import os
import json
from glob import glob
import PinataPy

pinataPy = PinataPy.PinataPy()

root_path = '/Users/yeznable/NCPW/'
dir_path = root_path + 'nft-metadata/'
paths_to_file = glob(dir_path + '*.json')
res = pinataPy.pin_file_to_ipfs(paths_to_file=paths_to_file)
IpfsHash = res['IpfsHash']

for file in paths_to_file:
    file_name = file[len(dir_path):]

    command = f'mv {dir_path}{file_name} {dir_path}done/{file_name}'
    # os.system(command)
    print(command)

with open('/Users/yeznable/NCPW/nft/scripts/mint-nft.js', 'r') as mint_nft_js:
    mint_nft = mint_nft_js.read().split('//step 5: Call the mintNFT function')[0]
