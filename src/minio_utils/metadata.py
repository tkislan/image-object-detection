import re

r = re.compile('x-amz-meta-(.*)', re.IGNORECASE)


def normalize_metadata(metadata: dict) -> dict:
    new_metadata = {}
    for meta_key in metadata.keys():
        m = r.match(meta_key)
        if not m:
            print('Invalid metadata key: {}'.format(meta_key))
            continue

        new_metadata[m.group(1).lower()] = metadata[meta_key]

    return new_metadata
