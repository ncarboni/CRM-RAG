"""Export RDF triples from SPARQL as GEXF for Gephi / Gephi Lite.

Usage:
    python export_gexf.py [--endpoint URL] [--output FILE]

Produces a GEXF file with:
  - Nodes colored by CRM type (FC-like grouping)
  - Node labels from rdfs:label
  - Edge labels from predicate local names
  - Node size proportional to degree
"""

import argparse
import json
import urllib.request
import urllib.parse
from collections import defaultdict
from xml.sax.saxutils import escape

PREFIXES = {
    "http://www.cidoc-crm.org/cidoc-crm/": "crm:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2002/07/owl#": "owl:",
    "http://w3id.org/vir#": "vir:",
}

SKIP_PREDICATES = {
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#inverseOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
}

# Type keyword → (group name, hex color)
TYPE_GROUPS = {
    "Human-Made": ("Thing", "#4FC3F7"),
    "Physical": ("Thing", "#4FC3F7"),
    "Persistent Item": ("Thing", "#4FC3F7"),
    "Iconographical": ("Thing", "#4FC3F7"),
    "Actor": ("Actor", "#FF8A65"),
    "Character": ("Actor", "#FF8A65"),
    "Place": ("Place", "#81C784"),
    "Activity": ("Event", "#BA68C8"),
    "Production": ("Event", "#BA68C8"),
    "Modification": ("Event", "#BA68C8"),
    "Event": ("Event", "#BA68C8"),
    "Beginning": ("Event", "#BA68C8"),
    "Type": ("Concept", "#FFD54F"),
    "Time-Span": ("Time", "#F06292"),
    "Visual Item": ("Visual", "#26C6DA"),
    "Representation": ("Visual", "#26C6DA"),
}


def shorten(uri):
    for prefix, short in PREFIXES.items():
        if uri.startswith(prefix):
            return short + uri[len(prefix):]
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def sparql_query(endpoint, query):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        endpoint, data=data,
        headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def classify_type(type_uris):
    for t in type_uris:
        short = shorten(t)
        for keyword, (group, color) in TYPE_GROUPS.items():
            if keyword in short:
                return group, color
    return "Other", "#8b949e"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:3030/asinou/sparql")
    parser.add_argument("--output", default="visualize_rdf_temp/asinou_rdf.gexf")
    args = parser.parse_args()

    print(f"Fetching types from {args.endpoint}...")
    rows = sparql_query(args.endpoint,
        "SELECT ?s ?type WHERE { ?s a ?type }")["results"]["bindings"]
    entity_types = defaultdict(list)
    for r in rows:
        entity_types[r["s"]["value"]].append(r["type"]["value"])
    print(f"  {len(entity_types)} typed entities")

    print("Fetching labels...")
    rows = sparql_query(args.endpoint,
        "SELECT ?s ?label WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label }"
    )["results"]["bindings"]
    labels = {}
    for r in rows:
        labels[r["s"]["value"]] = r["label"]["value"]
    print(f"  {len(labels)} labels")

    print("Fetching triples (URI→URI edges only)...")
    rows = sparql_query(args.endpoint, """
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            FILTER(isIRI(?o))
        }
    """)["results"]["bindings"]
    print(f"  {len(rows)} raw triples")

    # Build nodes and edges
    nodes = {}  # uri -> {label, group, color}
    edges = []
    degree = defaultdict(int)

    for r in rows:
        s, p, o = r["s"]["value"], r["p"]["value"], r["o"]["value"]
        if p in SKIP_PREDICATES:
            continue

        for uri in (s, o):
            if uri not in nodes:
                group, color = classify_type(entity_types.get(uri, []))
                nodes[uri] = {
                    "label": labels.get(uri, shorten(uri)),
                    "group": group,
                    "color": color,
                }

        degree[s] += 1
        degree[o] += 1
        edges.append((s, o, shorten(p)))

    print(f"  {len(nodes)} nodes, {len(edges)} edges after filtering")

    # Write GEXF
    print(f"Writing GEXF → {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gexf xmlns="http://gexf.net/1.3" version="1.3">\n')
        f.write('  <meta><creator>CRM_RAG SPARQL Export</creator></meta>\n')
        f.write('  <graph defaultedgetype="directed">\n')

        # Node attributes
        f.write('    <attributes class="node">\n')
        f.write('      <attribute id="0" title="group" type="string"/>\n')
        f.write('      <attribute id="1" title="uri" type="string"/>\n')
        f.write('      <attribute id="2" title="types" type="string"/>\n')
        f.write('    </attributes>\n')

        # Edge attributes
        f.write('    <attributes class="edge">\n')
        f.write('      <attribute id="0" title="predicate" type="string"/>\n')
        f.write('    </attributes>\n')

        # Nodes
        f.write('    <nodes>\n')
        for uri, data in nodes.items():
            r, g, b = hex_to_rgb(data["color"])
            deg = degree.get(uri, 1)
            size = 3 + min(30, deg * 0.5)
            lbl = escape(data["label"])
            grp = escape(data["group"])
            types_str = escape(", ".join(shorten(t) for t in entity_types.get(uri, [])))
            node_id = escape(uri)

            f.write(f'      <node id="{node_id}" label="{lbl}">\n')
            f.write(f'        <attvalues>\n')
            f.write(f'          <attvalue for="0" value="{grp}"/>\n')
            f.write(f'          <attvalue for="1" value="{node_id}"/>\n')
            f.write(f'          <attvalue for="2" value="{types_str}"/>\n')
            f.write(f'        </attvalues>\n')
            f.write(f'        <viz:color xmlns:viz="http://gexf.net/1.3/viz" r="{r}" g="{g}" b="{b}"/>\n')
            f.write(f'        <viz:size xmlns:viz="http://gexf.net/1.3/viz" value="{size:.1f}"/>\n')
            f.write(f'      </node>\n')
        f.write('    </nodes>\n')

        # Edges
        f.write('    <edges>\n')
        for i, (s, o, pred) in enumerate(edges):
            f.write(f'      <edge id="{i}" source="{escape(s)}" target="{escape(o)}" label="{escape(pred)}">\n')
            f.write(f'        <attvalues><attvalue for="0" value="{escape(pred)}"/></attvalues>\n')
            f.write(f'      </edge>\n')
        f.write('    </edges>\n')

        f.write('  </graph>\n')
        f.write('</gexf>\n')

    print(f"Done. Open {args.output} in Gephi Lite (gephi.org/gephi-lite) or Gephi desktop.")


if __name__ == "__main__":
    main()
