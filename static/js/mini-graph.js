/**
 * mini-graph.js — D3.js force-directed subgraph for per-answer visualization.
 *
 * API:
 *   renderMiniGraph(containerEl, sources) → { highlightNode(uri), unhighlightNode(uri), dispose() }
 *
 * Builds a small graph from retrieved sources:
 *   - Nodes = retrieved entities
 *   - Edges = triples where both subject and object are retrieved entities
 *   - Ghost nodes = non-retrieved entities appearing in multiple sources' triples
 */
const MiniGraph = (() => {

    const FC_COLORS = {
        Thing:   '#4FC3F7',
        Actor:   '#FF8A65',
        Place:   '#81C784',
        Event:   '#BA68C8',
        Concept: '#FFD54F',
        Time:    '#F06292'
    };

    const DEFAULT_COLOR = '#B0BEC5';

    function fcColor(fc) {
        return FC_COLORS[fc] || DEFAULT_COLOR;
    }

    /**
     * Build graph data from sources array.
     * Returns { nodes: [...], links: [...] } or null if not enough data.
     */
    function buildGraphData(sources) {
        const entitySet = new Set();
        const entityMap = {};        // uri → { label, fc, pagerank }
        const linkMap = new Map();   // "s|o" → { source, target, predicate_label }
        const ghostCandidates = {};  // uri → Set of source indices that mention it

        sources.forEach((src, idx) => {
            const uri = src.entity_uri;
            entitySet.add(uri);
            entityMap[uri] = {
                id: uri,
                label: src.entity_label || uri.split('/').pop(),
                fc: src.fc || '',
                pagerank: src.pagerank || 0,
                isGhost: false,
                sourceIndex: idx
            };
        });

        // Collect edges between retrieved entities + ghost candidates
        sources.forEach((src, idx) => {
            if (!src.raw_triples) return;
            src.raw_triples.forEach(triple => {
                const s = triple.subject;
                const o = triple.object;
                const sIn = entitySet.has(s);
                const oIn = entitySet.has(o);

                if (sIn && oIn && s !== o) {
                    const key = `${s}|${o}`;
                    if (!linkMap.has(key)) {
                        linkMap.set(key, {
                            source: s,
                            target: o,
                            predicate_label: triple.predicate_label || triple.predicate.split('/').pop().split('#').pop()
                        });
                    }
                }

                // Track non-entity URIs that appear across multiple sources (ghost candidates)
                if (sIn && !oIn && o.startsWith('http')) {
                    if (!ghostCandidates[o]) ghostCandidates[o] = new Set();
                    ghostCandidates[o].add(idx);
                }
                if (oIn && !sIn && s.startsWith('http')) {
                    if (!ghostCandidates[s]) ghostCandidates[s] = new Set();
                    ghostCandidates[s].add(idx);
                }
            });
        });

        // Add ghost nodes if fewer than 3 inter-entity edges
        if (linkMap.size < 3) {
            // Sort ghost candidates by how many sources mention them
            const ghosts = Object.entries(ghostCandidates)
                .filter(([, srcSet]) => srcSet.size >= 2)
                .sort((a, b) => b[1].size - a[1].size)
                .slice(0, 5);

            ghosts.forEach(([uri]) => {
                entityMap[uri] = {
                    id: uri,
                    label: uri.split('/').pop().split('#').pop(),
                    fc: '',
                    pagerank: 0,
                    isGhost: true,
                    sourceIndex: -1
                };
                entitySet.add(uri);
            });

            // Re-scan triples for new edges involving ghost nodes
            sources.forEach(src => {
                if (!src.raw_triples) return;
                src.raw_triples.forEach(triple => {
                    const s = triple.subject;
                    const o = triple.object;
                    if (entitySet.has(s) && entitySet.has(o) && s !== o) {
                        const key = `${s}|${o}`;
                        if (!linkMap.has(key)) {
                            linkMap.set(key, {
                                source: s,
                                target: o,
                                predicate_label: triple.predicate_label || triple.predicate.split('/').pop().split('#').pop()
                            });
                        }
                    }
                });
            });
        }

        const nodes = Object.values(entityMap);
        const links = Array.from(linkMap.values());

        // Hide graph if fewer than 2 nodes or 0 edges
        if (nodes.length < 2 || links.length === 0) return null;

        return { nodes, links };
    }

    /**
     * Render the mini-graph into containerEl.
     * Returns controller object or null if graph has insufficient data.
     */
    function renderMiniGraph(containerEl, sources) {
        const data = buildGraphData(sources);
        if (!data) {
            containerEl.style.display = 'none';
            return null;
        }

        containerEl.style.display = '';
        containerEl.innerHTML = '';

        const rect = containerEl.getBoundingClientRect();
        const width = rect.width || 500;
        const height = 220;

        const svg = d3.select(containerEl)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`);

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 22)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-4L10,0L0,4')
            .attr('fill', '#bbb');

        // PageRank → node radius
        const prValues = data.nodes.filter(n => !n.isGhost).map(n => n.pagerank);
        const prMin = Math.min(...prValues, 0);
        const prMax = Math.max(...prValues, 0.001);
        function nodeRadius(d) {
            if (d.isGhost) return 7;
            return 10 + 6 * ((d.pagerank - prMin) / (prMax - prMin || 1));
        }

        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(90))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 8));

        // Edge lines
        const link = svg.append('g')
            .selectAll('line')
            .data(data.links)
            .join('line')
            .attr('stroke', '#ccc')
            .attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrow)');

        // Edge labels (hidden by default, show on hover)
        const linkLabel = svg.append('g')
            .selectAll('text')
            .data(data.links)
            .join('text')
            .text(d => d.predicate_label)
            .attr('font-size', '8px')
            .attr('fill', '#999')
            .attr('text-anchor', 'middle')
            .attr('opacity', 0)
            .style('pointer-events', 'none');

        // Node groups
        const node = svg.append('g')
            .selectAll('g')
            .data(data.nodes)
            .join('g')
            .attr('class', 'mini-graph-node')
            .attr('data-uri', d => d.id)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // Node circles
        node.append('circle')
            .attr('r', d => nodeRadius(d))
            .attr('fill', d => d.isGhost ? '#E0E0E0' : fcColor(d.fc))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('opacity', d => d.isGhost ? 0.6 : 1);

        // Node labels
        node.append('text')
            .text(d => {
                const lbl = d.label;
                return lbl.length > 18 ? lbl.substring(0, 16) + '...' : lbl;
            })
            .attr('dy', d => nodeRadius(d) + 13)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#555')
            .style('pointer-events', 'none');

        // Hover: show edge labels + full node label tooltip
        node.on('mouseenter', function(event, d) {
            d3.select(this).select('circle')
                .transition().duration(150)
                .attr('stroke', '#333').attr('stroke-width', 3);

            // Show edge labels for connected edges
            linkLabel.attr('opacity', l =>
                (l.source.id === d.id || l.target.id === d.id) ? 1 : 0
            );

            // Tooltip
            d3.select(this).append('title').text(d.label);
        }).on('mouseleave', function() {
            d3.select(this).select('circle')
                .transition().duration(150)
                .attr('stroke', '#fff').attr('stroke-width', 2);
            linkLabel.attr('opacity', 0);
            d3.select(this).select('title').remove();
        });

        // Click → dispatch entity-select event
        node.on('click', function(event, d) {
            if (d.isGhost) return;
            containerEl.dispatchEvent(new CustomEvent('entity-select', {
                bubbles: true,
                detail: { uri: d.id, label: d.label }
            }));
        });

        // Tick
        simulation.on('tick', () => {
            // Clamp nodes inside bounds
            data.nodes.forEach(d => {
                const r = nodeRadius(d);
                d.x = Math.max(r + 2, Math.min(width - r - 2, d.x));
                d.y = Math.max(r + 2, Math.min(height - r - 2, d.y));
            });

            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            linkLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2 - 4);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // Controller
        return {
            highlightNode(uri) {
                node.each(function(d) {
                    if (d.id === uri) {
                        d3.select(this).select('circle')
                            .transition().duration(200)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 3.5)
                            .attr('r', nodeRadius(d) + 3);
                    }
                });
            },
            unhighlightNode(uri) {
                node.each(function(d) {
                    if (d.id === uri) {
                        d3.select(this).select('circle')
                            .transition().duration(200)
                            .attr('stroke', '#fff')
                            .attr('stroke-width', 2)
                            .attr('r', nodeRadius(d));
                    }
                });
            },
            dispose() {
                simulation.stop();
                containerEl.innerHTML = '';
            }
        };
    }

    return { renderMiniGraph };
})();
