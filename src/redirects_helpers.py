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

def per_project_filt(df, projects):

    df = df.filter(~df.page.contains('-') & (df.counts >= 1))

    if 'en' in projects:
        df = df.filter(~df.page.contains('User:') & \
                                 ~df.page.contains('Wikipedia:') & \
                                 ~df.page.contains('File:') & \
                                 ~df.page.contains('MediaWiki:') & \
                                 ~df.page.contains('Template:') & \
                                 ~df.page.contains('Help:') & \
                                 ~df.page.contains('Category:') & \
                                 ~df.page.contains('Portal:') & \
                                 ~df.page.contains('Draft:') & \
                                 ~df.page.contains('TimedText:') & \
                                 ~df.page.contains('Module:') & \
                                 ~df.page.contains('Special:') & \
                                 ~df.page.contains('Media:') & \
                                 ~df.page.contains('Talk:') & \
                                 ~df.page.contains('talk:') & \
                                 ~df.page.isin(specials_to_filt) \
                                 & (df.counts >= 1))
    elif 'fr' in projects:
        df = df.filter(~df.page.contains('Utilisateur:') & \
                                 ~df.page.contains('Wikipédia:') & \
                                 ~df.page.contains('Fichier:') & \
                                 ~df.page.contains('MediaWiki:') & \
                                 ~df.page.contains('Modèle:') & \
                                 ~df.page.contains('Aide:') & \
                                 ~df.page.contains('Catégorie:') & \
                                 ~df.page.contains('Portail:') & \
                                 ~df.page.contains('Projet:') & \
                                 ~df.page.contains('TimedText') & \
                                 ~df.page.contains('Référence:') & \
                                 ~df.page.contains('Module:') & \
                                 ~df.page.contains('Gadget:') & \
                                 ~df.page.contains('Sujet:') & \
                                 ~df.page.contains('Discussion') & \
                                 ~df.page.contains('Spécial') & \
                                 ~df.page.isin(specials_to_filt) \
                                 & (df.counts >= 1))

    elif 'ru' in projects:
        df = df.filter(~df.page.contains('Участник:') & \
                                 ~df.page.contains('Википедия:') & \
                                 ~df.page.contains('Файл:') & \
                                 ~df.page.contains('Шаблон:') & \
                                 ~df.page.contains('MediaWiki:') & \
                       ~df.page.contains('TimedText:') & \
                       ~df.page.contains('Справка:') & \
                                 ~df.page.contains('Категория:') & \
                                 ~df.page.contains('Портал:') & \
                                 ~df.page.contains('Инкубатор:') & \
                                 ~df.page.contains('Проект:') & \
                                 ~df.page.contains('Арбитраж:') & \
                                 ~df.page.contains('Модуль:') & \
                                 ~df.page.contains('Гаджет:') & \
                                 ~df.page.contains('Обсуждение'))

    elif 'es' in projects:
        df = df.filter(~df.page.contains('Discusión:') & \
                                 ~df.page.contains('discusión:') & \
                                 ~df.page.contains('Wikipedia:') & \
                                 ~df.page.contains('Usuario:') & \
                                 ~df.page.contains('MediaWiki:') & \
                                 ~df.page.contains('Archivo:') & \
                                 ~df.page.contains('Ayuda:') & \
                                 ~df.page.contains('Categoría:') & \
                                 ~df.page.contains('Portal:') & \
                                 ~df.page.contains('Módulo:') & \
                                 ~df.page.contains('Plantilla:') & \
                                 ~df.page.contains('Wikiproyecto:') & \
                                 ~df.page.contains('Anexo:') & \
                                 ~df.page.contains('Especial:') & \
                                 ~df.page.contains('Accesorio') & \
                                 ~df.page.contains('Medio:'))

    elif 'pt' in projects:
        df = df.filter(~df.page.contains('Discussão:') & \
                                 ~df.page.contains('discussão:') & \
                                 ~df.page.contains('Wikipédia:') & \
                                 ~df.page.contains('Especial:') & \
                                 ~df.page.contains('Usuário') & \
                                 ~df.page.contains('MediaWiki:') & \
                                 ~df.page.contains('Imagem:') & \
                                 ~df.page.contains('Ajuda:') & \
                                 ~df.page.contains('Categoria:') & \
                                 ~df.page.contains('Predefinição:') & \
                                 ~df.page.contains('Módulo:') & \
                                 ~df.page.contains('Ficheiro:') & \
                                 ~df.page.contains('Portal:') & \
                                 ~df.page.contains('Livro:') & \
                                 ~df.page.contains('talk:') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('TimedText') & \
                       ~df.page.contains('Education_Programm') & \
                       ~df.page.contains('Tópico:'))

    elif 'ar' in projects:
        df = df.filter(~df.page.contains('نقاش') & \
                                 ~df.page.contains('مستخدم') & \
                                 ~df.page.contains('ويكيبيديا') & \
                                 ~df.page.contains('ملف') & \
                                 ~df.page.contains('ميدياويكي') & \
                                 ~df.page.contains('قالب') & \
                                 ~df.page.contains('مساعدة') & \
                                 ~df.page.contains('تصنيف') & \
                                 ~df.page.contains('بوابة') & \
                                 ~df.page.contains('TimedText') & \
                                 ~df.page.contains('وحدة') & \
                                 ~df.page.contains('إضافة'))

    elif 'hu' in projects:
        df = df.filter(~df.page.contains('Vita:') & \
                                 ~df.page.contains('vita:') & \
                                 ~df.page.contains('Wikipédia:') & \
                                 ~df.page.contains('Szerkesztő:') & \
                                 ~df.page.contains('Fájl') & \
                                 ~df.page.contains('MediaWiki:') & \
                                 ~df.page.contains('Sablon:') & \
                                 ~df.page.contains('Segítség:') & \
                                 ~df.page.contains('Kategória:') & \
                                 ~df.page.contains('Szál:') & \
                                 ~df.page.contains('Összefoglaló:') & \
                                 ~df.page.contains('Portál:') & \
                                 ~df.page.contains('Cikkjelölt:') & \
                                 ~df.page.contains('Modul:') & \
                                 ~df.page.contains('talk:') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('TimedText'))

    elif 'fi' in projects:
        df = df.filter(~df.page.contains('Keskustelu') & \
                                 ~df.page.contains('vita:') & \
                                 ~df.page.contains('Wikipedia') & \
                                 ~df.page.contains('Käyttäjä:') & \
                                 ~df.page.contains('Tiedosto') & \
                                 ~df.page.contains('Järjestelmäviesti') & \
                                 ~df.page.contains('Malline') & \
                                 ~df.page.contains('Ohje') & \
                                 ~df.page.contains('Luokka:') & \
                                 ~df.page.contains('Teemasivu:') & \
                                 ~df.page.contains('Metasivu:') & \
                                 ~df.page.contains('Kirja:') & \
                                 ~df.page.contains('Wikiprojekti:') & \
                                 ~df.page.contains('Moduuli:') & \
                                 ~df.page.contains('talk:') & \
                       ~df.page.contains('Pienoisohjelma') & \
                       ~df.page.contains('TimedText'))

    elif 'uz' in projects:
        df = df.filter(~df.page.contains('Munozara') & \
                                 ~df.page.contains('munozara') & \
                                 ~df.page.contains('Foydalanuvchi') & \
                                 ~df.page.contains('Vikipediya') & \
                                 ~df.page.contains('Fayl') & \
                                 ~df.page.contains('MediaWiki') & \
                                 ~df.page.contains('Andoza') & \
                                 ~df.page.contains('Yordam') & \
                                 ~df.page.contains('Turkum:') & \
                                 ~df.page.contains('Portal:') & \
                                 ~df.page.contains('Loyiha:') & \
                                 ~df.page.contains('Kirja:') & \
                                 ~df.page.contains('Modul:') & \
                                 ~df.page.contains('talk:') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('TimedText'))

    elif 'af' in projects:
        df = df.filter(~df.page.contains('Bespreking') & \
                                 ~df.page.contains('bespreking') & \
                                 ~df.page.contains('Gebruiker') & \
                                 ~df.page.contains('Wikipedia') & \
                                 ~df.page.contains('Lêer') & \
                                 ~df.page.contains('MediaWiki') & \
                                 ~df.page.contains('Sjabloon') & \
                                 ~df.page.contains('Hulp') & \
                                 ~df.page.contains('Kategorie:') & \
                                 ~df.page.contains('Portaal:') & \
                                 ~df.page.contains('Module:') & \
                                 ~df.page.contains('talk:') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('TimedText'))

    elif 'my' in projects:
        df = df.filter(~df.page.contains('ဆွေးနွေးချက်') & \
                                 ~df.page.contains('အသုံးပြုသူ') & \
                                 ~df.page.contains('ဝီကီပီးဒီးယား') & \
                                 ~df.page.contains('ဖိုင်') & \
                                 ~df.page.contains('မီဒီယာဝီကီ') & \
                                 ~df.page.contains('တမ်းပလိတ်') & \
                                 ~df.page.contains('အကူအညီ') & \
                                 ~df.page.contains('ကဏ္') & \
                                 ~df.page.contains('မော်ဂျူ') & \
                                 ~df.page.contains('ကိရိယာငယ်') & \
                       ~df.page.contains('TimedText'))

    elif 'new' in projects:
        df = df.filter(~df.page.contains('खँलाबँला') & \
                                 ~df.page.contains('छ्येलेमि') & \
                                 ~df.page.contains('खँलाबँला') & \
                                 ~df.page.contains('खँलाबँला') & \
                                 ~df.page.contains('किपा') & \
                                 ~df.page.contains('मिडियाविकि') & \
                                 ~df.page.contains('ग्वाहालि') & \
                                 ~df.page.contains('पुचः') & \
                                 ~df.page.contains('दबू') & \
                       ~df.page.contains('TimedText') & \
                       ~df.page.contains('Module') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('talk') & \
                       ~df.page.contains('Template')
                       )
    elif 'pms' in projects:
        df = df.filter(~df.page.contains('Discussion') & \
                                 ~df.page.contains('Utent') & \
                                 ~df.page.contains('Ciaciarade') & \
                                 ~df.page.contains('Wikipedia') & \
                                 ~df.page.contains('Figura') & \
                                 ~df.page.contains('MediaWiki') & \
                                 ~df.page.contains('Stamp') & \
                                 ~df.page.contains('Agiut') & \
                                 ~df.page.contains('Categorìa') & \
                       ~df.page.contains('TimedText') & \
                       ~df.page.contains('Modulo') & \
                       ~df.page.contains('Accessorio') & \
                       ~df.page.contains('Definizione')
                       )

    elif 'jv' in projects:
        df = df.filter(~df.page.contains('Parembugan') & \
                                 ~df.page.contains('Naraguna') & \
                                 ~df.page.contains('Barkas') & \
                                 ~df.page.contains('Wikipédia') & \
                                 ~df.page.contains('MédhiaWiki') & \
                                 ~df.page.contains('Cithakan') & \
                                 ~df.page.contains('Pitulung') & \
                                 ~df.page.contains('Kategori') & \
                                 ~df.page.contains('Gapura') & \
                       ~df.page.contains('TimedText') & \
                       ~df.page.contains('Modhul') & \
                       ~df.page.contains('Gadget') & \
                       ~df.page.contains('talk')
                       )
    elif 'pnb' in projects:
        df = df.filter(~df.page.contains('گل بات') & \
                                 ~df.page.contains('ورتنوالا') & \
                                 ~df.page.contains('وکیپیڈیا') & \
                                 ~df.page.contains('فائل') & \
                                 ~df.page.contains('میڈیا وکی') & \
                                 ~df.page.contains('سانچہ') & \
                                 ~df.page.contains('ہتھونڈائی') & \
                                 ~df.page.contains('گٹھ') & \
                                 ~df.page.contains('ماڈیول') & \
                       ~df.page.contains('TimedText') & \
                       ~df.page.contains('آلہ')
                       )

    return df
