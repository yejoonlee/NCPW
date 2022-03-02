import os
import json
from glob import glob
import PinataPy

pinataPy = PinataPy.PinataPy()

# os.chdir('/Users/yeznable/NCPW/nft')

root_path = '/Users/yeznable/NCPW/'
dir_path = root_path + 'nft-img/'
paths_to_file = glob(dir_path + '*.png')
res = pinataPy.pin_file_to_ipfs(paths_to_file=paths_to_file)
print(res)
IpfsHash = res['IpfsHash']

for file in paths_to_file:
    file_name = file[len(dir_path):]

    tokenId = len(glob(dir_path + '/done/*.png'))
    item_name = file_name.split('_')[0]
    style = file_name.split('_')[1].split('.')[0]
    metadata = {
        "image": f"https://gateway.pinata.cloud/ipfs/{IpfsHash}{dir_path[6:]}{file_name}",
        "tokenId": tokenId,
        "name": f"NCPW {tokenId}",
        "attributes": [
            {
                "trait_type": "item_name",
                "value": item_name
            },
            {
                "trait_type": "style",
                "value": style
            }
        ]
    }

    with open(f'{root_path}/nft-metadata/{tokenId}.json', 'w') as outfile:
        json.dump(metadata, outfile, indent=4)

    command = f'mv {dir_path}{file_name} {dir_path}done/{file_name}'
    os.system(command)
    # print(command)

dir_path = root_path + 'nft-metadata/'
paths_to_file = glob(dir_path + '*.json')
res = pinataPy.pin_file_to_ipfs(paths_to_file=paths_to_file)
print(res)
IpfsHash = res['IpfsHash']

metadatas = []
for file in paths_to_file:
    file_name = file[len(dir_path):]
    metadatas.append(file_name)

    command = f'mv {dir_path}{file_name} {dir_path}done/{file_name}'
    os.system(command)
    # print(command)

metadata_root = f"https://gateway.pinata.cloud/ipfs/{IpfsHash}{dir_path[6:]}"
os.chdir('/Users/yeznable/NCPW/nft')

for f in metadatas:
    mint_nft_links = \
    f"""//step 5: Call the mintNFT function
mintNFT("{metadata_root + f}")
    """
    print(mint_nft_links)

    with open('/Users/yeznable/NCPW/nft/scripts/mint-nft.js', 'r') as mint_nft_js:
        mint_nft = mint_nft_js.read().split('//step 5: Call the mintNFT function')[0]

    with open('/Users/yeznable/NCPW/nft/scripts/mint-nft.js', 'w') as mint_nft_js:
        mint_nft = mint_nft + mint_nft_links
        mint_nft_js.write(mint_nft)

    os.system('node scripts/mint-nft.js')