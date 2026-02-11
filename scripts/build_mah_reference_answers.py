#!/usr/bin/env python3
"""
Build reference answers for MAH evaluation questions by querying the SPARQL endpoint.
"""

import json
import requests
from typing import Dict, List, Any
from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT = "http://localhost:3030/MAH/sparql"

def query_sparql(query: str) -> Dict[str, Any]:
    """Execute SPARQL query and return results."""
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def extract_bindings(results: Dict) -> List[Dict]:
    """Extract bindings from SPARQL results."""
    return results.get("results", {}).get("bindings", [])

def get_value(binding: Dict, key: str) -> str:
    """Extract value from binding."""
    return binding.get(key, {}).get("value", "")

# Q3: Ferdinand Hodler info and works
def q3_hodler_info():
    """Get info about Ferdinand Hodler and his paintings."""
    hodler_uri = "https://data.mahmah.ch/agent/8679"

    # Get Hodler's works via edition activities
    # The pattern is: edition P14_carried_out_by agent, and work URI is parent of edition URI
    query = f"""
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?work ?workLabel ?type ?typeLabel
    WHERE {{
        ?edition crm:P14_carried_out_by <{hodler_uri}> .
        BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
        ?work rdfs:label ?workLabel .
        OPTIONAL {{
            ?work crm:P2_has_type ?type .
            ?type rdfs:label ?typeLabel .
            FILTER(CONTAINS(LCASE(?typeLabel), "peinture") || CONTAINS(LCASE(?typeLabel), "painting"))
        }}
    }}
    LIMIT 200
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    works = []
    for b in bindings:
        work = {
            "uri": get_value(b, "work"),
            "label": get_value(b, "workLabel"),
            "type": get_value(b, "typeLabel")
        }
        if work not in works:  # Deduplicate
            works.append(work)

    return {
        "artist_uri": hodler_uri,
        "artist_name": "Ferdinand Hodler",
        "works_count": len(works),
        "works": works[:50]  # Limit for readability
    }

# Q4: Guerrier au morgenstern details
def q4_guerrier_info():
    """Get info about Guerrier au morgenstern."""
    guerrier_uri = "https://data.mahmah.ch/work/49580"

    query = f"""
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?p ?pLabel ?o ?oLabel
    WHERE {{
        <{guerrier_uri}> ?p ?o .
        OPTIONAL {{ ?p rdfs:label ?pLabel }}
        OPTIONAL {{ ?o rdfs:label ?oLabel }}
    }}
    LIMIT 200
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    properties = []
    for b in bindings:
        prop = {
            "predicate": get_value(b, "p"),
            "predicate_label": get_value(b, "pLabel"),
            "value": get_value(b, "o"),
            "value_label": get_value(b, "oLabel")
        }
        properties.append(prop)

    return {
        "work_uri": guerrier_uri,
        "work_label": "Guerrier au morgenstern",
        "properties": properties
    }

# Q5, Q6: Guerrier exhibitions
def q5_q6_guerrier_exhibitions():
    """Get exhibitions for Guerrier au morgenstern."""
    guerrier_uri = "https://data.mahmah.ch/work/49580"

    query = f"""
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?exhibition ?exhibitionLabel ?place ?placeLabel ?timespan ?start ?end
    WHERE {{
        <{guerrier_uri}> crm:P16i_was_used_for ?exhibition .
        ?exhibition rdfs:label ?exhibitionLabel .
        OPTIONAL {{
            ?exhibition crm:P7_took_place_at ?place .
            ?place rdfs:label ?placeLabel .
        }}
        OPTIONAL {{
            ?exhibition crm:P4_has_time-span ?timespan .
            OPTIONAL {{ ?timespan crm:P81a_end_of_the_begin ?start }}
            OPTIONAL {{ ?timespan crm:P81b_begin_of_the_end ?end }}
        }}
    }}
    ORDER BY ?start
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    exhibitions = []
    for b in bindings:
        exh = {
            "uri": get_value(b, "exhibition"),
            "label": get_value(b, "exhibitionLabel"),
            "place": get_value(b, "placeLabel"),
            "start_date": get_value(b, "start"),
            "end_date": get_value(b, "end")
        }
        exhibitions.append(exh)

    # For each exhibition, get other works
    for exh in exhibitions:
        if "Gianadda" in exh["label"] or "gianadda" in exh["label"].lower():
            # Get other works in this exhibition
            query2 = f"""
            PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?work ?workLabel
            WHERE {{
                ?work crm:P16i_was_used_for <{exh["uri"]}> .
                ?work rdfs:label ?workLabel .
                FILTER(?work != <{guerrier_uri}>)
            }}
            LIMIT 50
            """
            results2 = query_sparql(query2)
            bindings2 = extract_bindings(results2)
            exh["other_works"] = [{"uri": get_value(b, "work"), "label": get_value(b, "workLabel")}
                                   for b in bindings2]

    return {
        "work_uri": guerrier_uri,
        "exhibitions": exhibitions
    }

# Q7, Q8: Hodler exhibitions
def q7_q8_hodler_exhibitions():
    """Get exhibitions featuring Hodler's works."""
    hodler_uri = "https://data.mahmah.ch/agent/8679"

    query = f"""
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?exhibition ?exhibitionLabel ?place ?placeLabel ?start ?end
    WHERE {{
        ?edition crm:P14_carried_out_by <{hodler_uri}> .
        BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
        ?work crm:P16i_was_used_for ?exhibition .
        ?exhibition rdfs:label ?exhibitionLabel .
        OPTIONAL {{
            ?exhibition crm:P7_took_place_at ?place .
            ?place rdfs:label ?placeLabel .
        }}
        OPTIONAL {{
            ?exhibition crm:P4_has_time-span ?timespan .
            ?timespan crm:P81a_end_of_the_begin ?start .
            ?timespan crm:P81b_begin_of_the_end ?end .
        }}
    }}
    ORDER BY ?start
    LIMIT 100
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    exhibitions = []
    for b in bindings:
        exh = {
            "uri": get_value(b, "exhibition"),
            "label": get_value(b, "exhibitionLabel"),
            "place": get_value(b, "placeLabel"),
            "start_date": get_value(b, "start"),
            "end_date": get_value(b, "end")
        }
        exhibitions.append(exh)

    return {
        "artist_uri": hodler_uri,
        "exhibitions_count": len(exhibitions),
        "exhibitions": exhibitions
    }

# Q1: Swiss artists' works
def q1_swiss_artists():
    """Find works by Swiss artists in the collection."""
    # Search for artists born/active in Swiss places
    query_swiss_artists = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>

    SELECT DISTINCT ?agent ?agentLabel (COUNT(DISTINCT ?work) as ?workCount)
    WHERE {
        ?agent a crm:E39_Actor .
        ?agent rdfs:label ?agentLabel .
        ?agent crm:P92i_was_brought_into_existence_by ?birth .
        ?birth crm:P7_took_place_at ?place .
        ?place rdfs:label ?placeLabel .
        FILTER(CONTAINS(LCASE(?placeLabel), "genève") ||
               CONTAINS(LCASE(?placeLabel), "geneva") ||
               CONTAINS(LCASE(?placeLabel), "suisse") ||
               CONTAINS(LCASE(?placeLabel), "switzerland") ||
               CONTAINS(LCASE(?placeLabel), "bern") ||
               CONTAINS(LCASE(?placeLabel), "zürich") ||
               CONTAINS(LCASE(?placeLabel), "lausanne") ||
               CONTAINS(LCASE(?placeLabel), "basel"))
        ?edition crm:P14_carried_out_by ?agent .
        BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
    }
    GROUP BY ?agent ?agentLabel
    ORDER BY DESC(?workCount)
    LIMIT 100
    """

    results = query_sparql(query_swiss_artists)
    bindings = extract_bindings(results)

    artists = []
    for b in bindings:
        artist = {
            "uri": get_value(b, "agent"),
            "label": get_value(b, "agentLabel"),
            "work_count": int(get_value(b, "workCount") or 0)
        }
        artists.append(artist)

    return {
        "method": "birthplace_search",
        "artists_count": len(artists),
        "total_works": sum(a["work_count"] for a in artists),
        "artists": artists[:50]  # Top 50 Swiss artists
    }

# Q2: Paintings depicting Geneva
def q2_geneva_paintings():
    """Find paintings that depict Geneva."""
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    PREFIX vir: <http://w3id.org/vir#>

    SELECT DISTINCT ?work ?workLabel ?depicted ?depictedLabel
    WHERE {
        # Find Geneva as a subject
        ?depicted rdfs:label ?depictedLabel .
        FILTER(CONTAINS(LCASE(?depictedLabel), "genève") || CONTAINS(LCASE(?depictedLabel), "geneva"))

        # Works depicting Geneva
        {
            ?work crm:P62_depicts ?depicted .
            ?work rdfs:label ?workLabel .
        }
        UNION
        {
            ?work vir:K24_portray ?depicted .
            ?work rdfs:label ?workLabel .
        }

        # Is a painting
        ?work crm:P2_has_type ?type .
        ?type rdfs:label ?typeLabel .
        FILTER(CONTAINS(LCASE(?typeLabel), "peinture") || CONTAINS(LCASE(?typeLabel), "painting"))
    }
    LIMIT 200
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    paintings = []
    for b in bindings:
        painting = {
            "uri": get_value(b, "work"),
            "label": get_value(b, "workLabel"),
            "depicts": get_value(b, "depictedLabel")
        }
        paintings.append(painting)

    return {
        "query_type": "depicts_geneva",
        "paintings_count": len(paintings),
        "paintings": paintings
    }

# Q9: Other Swiss artists
def q9_other_swiss_artists():
    """Find other Swiss artists besides Hodler."""
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>

    SELECT DISTINCT ?agent ?agentLabel (COUNT(DISTINCT ?work) as ?workCount)
    WHERE {
        ?agent a crm:E39_Actor .
        ?agent rdfs:label ?agentLabel .
        ?agent crm:P92i_was_brought_into_existence_by ?birth .
        ?birth crm:P7_took_place_at ?place .
        ?place rdfs:label ?placeLabel .
        FILTER(CONTAINS(LCASE(?placeLabel), "genève") ||
               CONTAINS(LCASE(?placeLabel), "geneva") ||
               CONTAINS(LCASE(?placeLabel), "suisse") ||
               CONTAINS(LCASE(?placeLabel), "switzerland") ||
               CONTAINS(LCASE(?placeLabel), "bern") ||
               CONTAINS(LCASE(?placeLabel), "zürich") ||
               CONTAINS(LCASE(?placeLabel), "lausanne") ||
               CONTAINS(LCASE(?placeLabel), "basel"))
        ?edition crm:P14_carried_out_by ?agent .
        BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
        FILTER(?agent != <https://data.mahmah.ch/agent/8679>)
    }
    GROUP BY ?agent ?agentLabel
    ORDER BY DESC(?workCount)
    LIMIT 50
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    artists = []
    for b in bindings:
        artist = {
            "uri": get_value(b, "agent"),
            "label": get_value(b, "agentLabel"),
            "work_count": int(get_value(b, "workCount") or 0)
        }
        artists.append(artist)

    return {
        "query_type": "swiss_artists_excluding_hodler",
        "artists_count": len(artists),
        "artists": artists
    }

# Q10: Hans Schweizer
def q10_hans_schweizer():
    """Get info about Hans Schweizer."""
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>

    SELECT DISTINCT ?agent ?agentLabel
    WHERE {
        ?agent rdfs:label ?agentLabel .
        ?agent a crm:E39_Actor .
        FILTER(CONTAINS(LCASE(?agentLabel), "schweizer"))
    }
    LIMIT 20
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    schweizers = []
    for b in bindings:
        schweizer = {
            "uri": get_value(b, "agent"),
            "label": get_value(b, "agentLabel")
        }
        schweizers.append(schweizer)

    # For each Schweizer, get their works
    for schweizer in schweizers:
        if "Hans" in schweizer["label"]:
            query2 = f"""
            PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?work ?workLabel ?type ?typeLabel
            WHERE {{
                ?edition crm:P14_carried_out_by <{schweizer["uri"]}> .
                BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
                ?work rdfs:label ?workLabel .
                OPTIONAL {{
                    ?work crm:P2_has_type ?type .
                    ?type rdfs:label ?typeLabel .
                }}
            }}
            LIMIT 100
            """
            results2 = query_sparql(query2)
            bindings2 = extract_bindings(results2)
            schweizer["works"] = [{"uri": get_value(b, "work"), "label": get_value(b, "workLabel"),
                                   "type": get_value(b, "typeLabel")} for b in bindings2]

    return {
        "schweizers_found": schweizers
    }

# Q11: Top 10 Swiss artists
def q11_top_swiss_artists():
    """Get top 10 Swiss artists by work count."""
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>

    SELECT DISTINCT ?agent ?agentLabel (COUNT(DISTINCT ?work) as ?workCount)
    WHERE {
        ?agent a crm:E39_Actor .
        ?agent rdfs:label ?agentLabel .
        ?agent crm:P92i_was_brought_into_existence_by ?birth .
        ?birth crm:P7_took_place_at ?place .
        ?place rdfs:label ?placeLabel .
        FILTER(CONTAINS(LCASE(?placeLabel), "genève") ||
               CONTAINS(LCASE(?placeLabel), "geneva") ||
               CONTAINS(LCASE(?placeLabel), "suisse") ||
               CONTAINS(LCASE(?placeLabel), "switzerland") ||
               CONTAINS(LCASE(?placeLabel), "bern") ||
               CONTAINS(LCASE(?placeLabel), "zürich") ||
               CONTAINS(LCASE(?placeLabel), "lausanne") ||
               CONTAINS(LCASE(?placeLabel), "basel"))
        ?edition crm:P14_carried_out_by ?agent .
        BIND(IRI(REPLACE(STR(?edition), "/edition", "")) AS ?work)
    }
    GROUP BY ?agent ?agentLabel
    ORDER BY DESC(?workCount)
    LIMIT 10
    """

    results = query_sparql(query)
    bindings = extract_bindings(results)

    artists = []
    for b in bindings:
        artist = {
            "uri": get_value(b, "agent"),
            "label": get_value(b, "agentLabel"),
            "work_count": int(get_value(b, "workCount") or 0)
        }
        artists.append(artist)

    return {
        "top_10_swiss_artists": artists
    }

def main():
    """Build all reference answers."""
    print("Building MAH reference answers...")

    reference_answers = {}

    print("\nQ1: Swiss artists' works...")
    try:
        reference_answers["Q1"] = q1_swiss_artists()
        print(f"  Found {reference_answers['Q1']['artists_count']} Swiss artists")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q1"] = {"error": str(e)}

    print("\nQ2: Paintings depicting Geneva...")
    try:
        reference_answers["Q2"] = q2_geneva_paintings()
        print(f"  Found {reference_answers['Q2']['paintings_count']} paintings")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q2"] = {"error": str(e)}

    print("\nQ3: Ferdinand Hodler info and works...")
    try:
        reference_answers["Q3"] = q3_hodler_info()
        print(f"  Found {reference_answers['Q3']['works_count']} works")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q3"] = {"error": str(e)}

    print("\nQ4: Guerrier au morgenstern details...")
    try:
        reference_answers["Q4"] = q4_guerrier_info()
        print(f"  Found {len(reference_answers['Q4']['properties'])} properties")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q4"] = {"error": str(e)}

    print("\nQ5-Q6: Guerrier exhibitions...")
    try:
        reference_answers["Q5_Q6"] = q5_q6_guerrier_exhibitions()
        print(f"  Found {len(reference_answers['Q5_Q6']['exhibitions'])} exhibitions")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q5_Q6"] = {"error": str(e)}

    print("\nQ7-Q8: Hodler exhibitions...")
    try:
        reference_answers["Q7_Q8"] = q7_q8_hodler_exhibitions()
        print(f"  Found {reference_answers['Q7_Q8']['exhibitions_count']} exhibitions")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q7_Q8"] = {"error": str(e)}

    print("\nQ9: Other Swiss artists...")
    try:
        reference_answers["Q9"] = q9_other_swiss_artists()
        print(f"  Found {reference_answers['Q9']['artists_count']} artists")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q9"] = {"error": str(e)}

    print("\nQ10: Hans Schweizer...")
    try:
        reference_answers["Q10"] = q10_hans_schweizer()
        print(f"  Found {len(reference_answers['Q10']['schweizers_found'])} Schweizers")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q10"] = {"error": str(e)}

    print("\nQ11: Top 10 Swiss artists...")
    try:
        reference_answers["Q11"] = q11_top_swiss_artists()
        print(f"  Top 10 artists retrieved")
    except Exception as e:
        print(f"  ERROR: {e}")
        reference_answers["Q11"] = {"error": str(e)}

    # Save to file
    output_file = "/Users/carboni/Documents/Academia/UIUC/Projects/RAG/Code/reports/mah_reference_answers.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reference_answers, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Reference answers saved to {output_file}")

if __name__ == "__main__":
    main()
