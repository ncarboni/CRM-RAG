"""Export RDF triples from a SPARQL endpoint as an interactive HTML graph.

Usage:
    python export_rdf_graph.py [--endpoint URL] [--output FILE] [--limit N]

Defaults to the asinou dataset on localhost Fuseki.
Produces a self-contained HTML file with Cytoscape.js visualization.
"""

import argparse
import json
import urllib.request
import urllib.parse
from collections import defaultdict

PREFIXES = {
    "http://www.cidoc-crm.org/cidoc-crm/": "crm:",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2002/07/owl#": "owl:",
    "http://w3id.org/vir#": "vir:",
    "http://www.w3.org/2004/02/skos/core#": "skos:",
}

# Predicates to skip (schema-level, not instance data)
SKIP_PREDICATES = {
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#inverseOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#comment",
}

# FC color mapping
FC_COLORS = {
    "E22_Human-Made_Object": "#4e79a7",
    "E39_Actor": "#f28e2b",
    "E53_Place": "#59a14f",
    "E7_Activity": "#e15759",
    "E55_Type": "#b07aa1",
    "E52_Time-Span": "#76b7b2",
}


def shorten_uri(uri):
    for prefix, short in PREFIXES.items():
        if uri.startswith(prefix):
            return short + uri[len(prefix):]
    # For data URIs, take last path segment
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def sparql_query(endpoint, query):
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        endpoint, data=data,
        headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch_triples(endpoint, limit=None):
    """Fetch all instance triples (skip schema and literals for graph edges)."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT ?s ?p ?o ?sLabel ?oLabel WHERE {{
        ?s ?p ?o .
        FILTER(isIRI(?o))
        OPTIONAL {{ ?s <http://www.w3.org/2000/01/rdf-schema#label> ?sLabel }}
        OPTIONAL {{ ?o <http://www.w3.org/2000/01/rdf-schema#label> ?oLabel }}
    }}
    {limit_clause}
    """
    result = sparql_query(endpoint, query)
    return result["results"]["bindings"]


def fetch_types(endpoint):
    """Fetch rdf:type for all entities."""
    query = """
    SELECT ?s ?type WHERE {
        ?s a ?type .
    }
    """
    result = sparql_query(endpoint, query)
    types = {}
    for row in result["results"]["bindings"]:
        uri = row["s"]["value"]
        t = row["type"]["value"]
        types[uri] = t
    return types


def build_graph(triples, entity_types):
    nodes = {}
    edges = []

    for row in triples:
        s = row["s"]["value"]
        p = row["p"]["value"]
        o = row["o"]["value"]

        if p in SKIP_PREDICATES:
            continue

        s_label = row.get("sLabel", {}).get("value", "")
        o_label = row.get("oLabel", {}).get("value", "")

        if s not in nodes:
            nodes[s] = {
                "id": s,
                "label": s_label or shorten_uri(s),
                "type": shorten_uri(entity_types.get(s, "")),
            }
        if o not in nodes:
            nodes[o] = {
                "id": o,
                "label": o_label or shorten_uri(o),
                "type": shorten_uri(entity_types.get(o, "")),
            }

        edges.append({
            "source": s,
            "target": o,
            "label": shorten_uri(p),
        })

    return list(nodes.values()), edges


def get_node_color(node_type):
    for crm_class, color in FC_COLORS.items():
        if crm_class in node_type:
            return color
    return "#999999"


def generate_html(nodes, edges, title="RDF Graph"):
    # Assign colors
    for n in nodes:
        n["color"] = get_node_color(n["type"])

    cy_nodes = json.dumps([
        {
            "data": {
                "id": n["id"],
                "label": n["label"][:40],
                "fullLabel": n["label"],
                "type": n["type"],
                "color": n["color"],
            }
        }
        for n in nodes
    ])

    cy_edges = json.dumps([
        {
            "data": {
                "source": e["source"],
                "target": e["target"],
                "label": e["label"],
            }
        }
        for e in edges
    ])

    legend_items = "".join(
        f'<span style="color:{c};font-weight:bold;margin-right:12px;">&#9679; {cls}</span>'
        for cls, c in FC_COLORS.items()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
  body {{ margin:0; font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; }}
  #cy {{ width:100%; height:calc(100vh - 90px); }}
  #controls {{ padding:8px 16px; background:#16213e; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  #controls input {{ padding:4px 8px; border-radius:4px; border:1px solid #555; background:#0f3460; color:#eee; }}
  #controls button {{ padding:4px 12px; border-radius:4px; border:none; background:#533483; color:#eee; cursor:pointer; }}
  #controls button:hover {{ background:#6a44a0; }}
  #legend {{ padding:4px 16px; background:#0f3460; font-size:12px; }}
  #info {{ position:fixed; bottom:16px; right:16px; background:#16213e; padding:12px; border-radius:8px;
           max-width:350px; font-size:12px; display:none; border:1px solid #533483; }}
  #info h4 {{ margin:0 0 6px; color:#e94560; }}
  #info .close {{ float:right; cursor:pointer; color:#e94560; }}
</style>
</head>
<body>
<div id="controls">
  <strong>{title}</strong>
  <span style="color:#888;">|</span>
  <span style="font-size:12px;">{len(nodes)} nodes, {len(edges)} edges</span>
  <span style="color:#888;">|</span>
  <input id="search" type="text" placeholder="Search nodes..." size="20">
  <button onclick="resetView()">Reset</button>
  <button onclick="toggleLabels()">Toggle Labels</button>
  <label style="font-size:12px;"><input type="checkbox" id="edgeLabels" onchange="toggleEdgeLabels()"> Edge labels</label>
</div>
<div id="legend">{legend_items} <span style="color:#999;">&#9679; Other</span></div>
<div id="cy"></div>
<div id="info"><span class="close" onclick="this.parentElement.style.display='none'">&#10005;</span><h4 id="infoTitle"></h4><div id="infoBody"></div></div>

<script>
var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: {{
    nodes: {cy_nodes},
    edges: {cy_edges}
  }},
  style: [
    {{ selector: 'node', style: {{
        'background-color': 'data(color)',
        'label': 'data(label)',
        'font-size': '8px',
        'color': '#ccc',
        'text-valign': 'bottom',
        'text-margin-y': 4,
        'width': 12,
        'height': 12,
    }}}},
    {{ selector: 'edge', style: {{
        'width': 1,
        'line-color': '#444',
        'target-arrow-color': '#666',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 0.6,
        'font-size': '6px',
        'color': '#888',
    }}}},
    {{ selector: '.highlighted', style: {{
        'background-color': '#e94560',
        'width': 20,
        'height': 20,
        'font-size': '11px',
        'color': '#fff',
        'z-index': 999,
    }}}},
    {{ selector: '.neighbor', style: {{
        'background-color': '#f7d060',
        'width': 16,
        'height': 16,
        'font-size': '10px',
        'color': '#fff',
    }}}},
    {{ selector: '.highlighted-edge', style: {{
        'line-color': '#e94560',
        'target-arrow-color': '#e94560',
        'width': 2,
        'label': 'data(label)',
    }}}},
    {{ selector: '.faded', style: {{
        'opacity': 0.15,
    }}}},
  ],
  layout: {{
    name: 'cose',
    animate: false,
    nodeDimensionsIncludeLabels: true,
    idealEdgeLength: function(edge) {{ return 100; }},
    nodeRepulsion: function(node) {{ return 8000; }},
    gravity: 0.3,
    numIter: 1000,
    randomize: true,
  }}
}});

var labelsVisible = true;
function toggleLabels() {{
  labelsVisible = !labelsVisible;
  cy.style().selector('node').style('label', labelsVisible ? 'data(label)' : '').update();
}}
function toggleEdgeLabels() {{
  var show = document.getElementById('edgeLabels').checked;
  cy.style().selector('edge').style('label', show ? 'data(label)' : '').update();
}}
function resetView() {{
  cy.elements().removeClass('highlighted neighbor highlighted-edge faded');
  document.getElementById('info').style.display = 'none';
  cy.fit();
}}

document.getElementById('search').addEventListener('input', function(e) {{
  var term = e.target.value.toLowerCase();
  cy.elements().removeClass('highlighted neighbor highlighted-edge faded');
  if (!term) return;
  var matched = cy.nodes().filter(n => n.data('label').toLowerCase().includes(term) || n.data('fullLabel').toLowerCase().includes(term));
  if (matched.length > 0) {{
    cy.elements().addClass('faded');
    matched.removeClass('faded').addClass('highlighted');
    matched.connectedEdges().removeClass('faded').addClass('highlighted-edge');
    matched.neighborhood().nodes().removeClass('faded').addClass('neighbor');
    if (matched.length <= 5) cy.fit(matched.union(matched.neighborhood()), 50);
  }}
}});

cy.on('tap', 'node', function(e) {{
  var node = e.target;
  cy.elements().removeClass('highlighted neighbor highlighted-edge faded');
  cy.elements().addClass('faded');
  node.removeClass('faded').addClass('highlighted');
  node.connectedEdges().removeClass('faded').addClass('highlighted-edge');
  node.neighborhood().nodes().removeClass('faded').addClass('neighbor');
  var edges = node.connectedEdges().map(e => {{
    var other = e.source().id() === node.id() ? e.target() : e.source();
    return '<b>' + e.data('label') + '</b> → ' + other.data('label');
  }});
  document.getElementById('infoTitle').textContent = node.data('fullLabel');
  document.getElementById('infoBody').innerHTML =
    '<p style="color:#888;">' + node.data('type') + '</p>' +
    '<p><b>URI:</b> <span style="font-size:10px;word-break:break-all;">' + node.id() + '</span></p>' +
    '<p><b>Edges (' + edges.length + '):</b></p>' +
    '<ul style="margin:0;padding-left:16px;max-height:200px;overflow-y:auto;">' +
    edges.map(e => '<li style="font-size:11px;">' + e + '</li>').join('') + '</ul>';
  document.getElementById('info').style.display = 'block';
}});

cy.on('tap', function(e) {{
  if (e.target === cy) resetView();
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Export RDF triples as interactive HTML graph")
    parser.add_argument("--endpoint", default="http://localhost:3030/asinou/sparql",
                        help="SPARQL endpoint URL")
    parser.add_argument("--output", default="visualize_rdf_temp/rdf_graph.html",
                        help="Output HTML file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of triples (default: all)")
    parser.add_argument("--title", default="Asinou RDF Graph",
                        help="Page title")
    args = parser.parse_args()

    print(f"Fetching types from {args.endpoint}...")
    entity_types = fetch_types(args.endpoint)
    print(f"  {len(entity_types)} typed entities")

    print(f"Fetching triples...")
    triples = fetch_triples(args.endpoint, limit=args.limit)
    print(f"  {len(triples)} triples fetched")

    print("Building graph...")
    nodes, edges = build_graph(triples, entity_types)
    print(f"  {len(nodes)} nodes, {len(edges)} edges")

    print(f"Generating HTML → {args.output}")
    html = generate_html(nodes, edges, title=args.title)
    with open(args.output, "w") as f:
        f.write(html)

    print(f"Done. Open {args.output} in a browser.")


if __name__ == "__main__":
    main()
