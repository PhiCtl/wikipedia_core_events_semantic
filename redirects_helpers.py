import requests
import numpy as np

def chunk_split(list_of_ids, chunk_len=49):
    """
    Split the given list into chunks
    :param : list_of_ids
    :param : chunk_len
    Usage : for Wikipedia API, max 50 approx. simultaneous requests
    """

    l = []
    for i in range(0, len(list_of_ids), chunk_len):
        l.append(list_of_ids[i:i + chunk_len])
    if len(l) * 49 < len(list_of_ids):
        l.append(list_of_ids[i:])
    return l


def yield_mapping(pages, prop='redirects', subprop='pageid'):
    """
    Parse API request response to get target page to ids mapping
    :param pages: part of API request response
    """
    mapping = {}

    # Collect all redirects ids
    for p_id, p in pages.items():
        if prop not in p:
            mapping[p_id] = p_id
        else:
            rids = [str(d[subprop]) for d in p[prop]]
            for r in rids:
                mapping[r] = p_id

    return mapping


def query_target_id(request, project):
    """
    Query Wikipedia API with specified parameters.
    Adapted From https://github.com/pgilders/WikiNewsNetwork-01-WikiNewsTopics
    Parameters
    ----------
    request : dict
        API call parameters.
    project : str
        project to query from
    Raises
    ------
    ValueError
        Raises error if returned by API.
    Yields
    ------
    dict
        Subsequent dicts of json API response.
    """

    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify with values from the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            f'https://{project}.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            print('ERROR')
            raise ValueError(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']['pages']
        if 'continue' not in result:
            break
        lastContinue = result['continue']


def get_target_id(ids, request_type='redirects', request_id='pageids', project='en'):
    """
    Map ids to their target page id
    :param ids: list of ids to match to target page id
    """

    chunk_list = chunk_split(ids)
    print(f"Matching {len(ids)} ids")
    mapping = {}

    for chunk in tqdm(chunk_list):
        params = {'action': 'query', 'format': 'json', request_id: '|'.join(chunk),
                  'prop': request_type}
        if request_type == 'redirects':
            params[request_type] = 'True'
            params['rdlimit'] = 'max'
        for res in query_target_id(params, project=project):
            m = yield_mapping(res, prop=request_type, subprop=request_id[:-1])
            mapping.update({k : v for k, v in m.items() if k in chunk})

    return mapping


def invert_mapping(inv_map, ids):
    """
    Invert mapping and select relevant keys
    """
    mapping = {v: k for k, vs in inv_map.items() for v in vs if v in ids}
    return mapping