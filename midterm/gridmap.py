import json
import os
import sys
import traceback
import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objs as go
import plotly.io as pio

def load_pose_graph(json_path):
    if not os.path.exists(json_path):
        sys.exit(1)
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            return data
    except Exception as e:
        sys.exit(1)

def parse_images_file(file_path):
    positions = {}
    orientations = {}
    if not os.path.exists(file_path):
        sys.exit(1)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            metadata = line.strip().split()
            if len(metadata) >= 8:
                try:
                    image_id = int(metadata[0])
                    qw, qx, qy, qz = map(float, metadata[1:5])
                    tx, ty, tz = map(float, metadata[5:8])
                    positions[image_id] = (tx, ty)
                    orientations[image_id] = (qw, qx, qy, qz)
                except:
                    pass
                i += 2
            else:
                i += 1
    return positions, orientations

def build_grid_map(positions, distance_threshold=30.0, max_neighbors=5, edges_file='edges.txt'):
    node_ids = list(positions.keys())
    pos_array = np.array(list(positions.values()))
    tree = cKDTree(pos_array)
    total_nodes = len(node_ids)
    try:
        with open(edges_file, 'w') as f:
            for idx in range(total_nodes):
                distances, indices = tree.query(
                    pos_array[idx],
                    k=max_neighbors + 1,
                    distance_upper_bound=distance_threshold,
                    workers=-1
                )
                for i in range(1, len(indices)):
                    neighbor_idx = indices[i]
                    distance = distances[i]
                    if neighbor_idx >= total_nodes or np.isinf(distance):
                        continue
                    node1 = node_ids[idx]
                    node2 = node_ids[neighbor_idx]
                    if node1 < node2:
                        f.write(f"{node1} {node2}\n")
    except Exception as e:
        sys.exit(1)

def build_graph_from_edges(combined_positions, edges_file='edges.txt', graph_file='graph.pickle'):
    graph = ig.Graph()
    node_ids = sorted(combined_positions.keys())
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    graph.add_vertices(len(node_ids))
    pos_array = [combined_positions[node_id] for node_id in node_ids]
    graph.vs['pos'] = pos_array
    graph.vs['id'] = node_ids
    edges = []
    try:
        with open(edges_file, 'r') as f:
            for line in f:
                try:
                    node1, node2 = map(int, line.strip().split())
                    idx1 = node_id_to_index[node1]
                    idx2 = node_id_to_index[node2]
                    edges.append((idx1, idx2))
                except:
                    pass
    except Exception as e:
        sys.exit(1)
    graph.add_edges(edges)
    try:
        graph.write_pickle(graph_file)
    except Exception as e:
        sys.exit(1)
    return graph, node_id_to_index

def set_start_end_nodes(graph, node_id_to_index, positions):
    index_positions = {node_id_to_index[node_id]: pos for node_id, pos in positions.items()}
    try:
        start_index = min(index_positions, key=lambda k: (index_positions[k][0], index_positions[k][1]))
        end_index = max(index_positions, key=lambda k: (index_positions[k][0], index_positions[k][1]))
    except ValueError as e:
        sys.exit(1)
    try:
        graph.vs[start_index]['type'] = 'start'
        graph.vs[end_index]['type'] = 'end'
    except Exception as e:
        sys.exit(1)
    return start_index, end_index

def visualize_complete_graph(graph, output_html='graph_visualization.html'):
    if 'pos' in graph.vs.attributes():
        positions = graph.vs['pos']
        x, y = zip(*positions)
    else:
        layout = graph.layout('fr')
        x, y = zip(*layout.coords)
    edge_x = []
    edge_y = []
    for edge in graph.es:
        source = edge.source
        target = edge.target
        edge_x += [x[source], x[target], None]
        edge_y += [y[source], y[target], None]
    node_x = x
    node_y = y
    node_colors = []
    for v in graph.vs:
        if 'type' in v.attributes() and v['type'] == 'start':
            node_colors.append('green')
        elif 'type' in v.attributes() and v['type'] == 'end':
            node_colors.append('red')
        else:
            node_colors.append('blue')
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=2,
            line=dict(width=0)
        )
    )
    node_text = []
    for v in graph.vs:
        text = f"ID: {v['id']}"
        if 'type' in v.attributes():
            text += f"<br>Type: {v['type']}"
        node_text.append(text)
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        width=2000,
                        height=2000,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    try:
        pio.write_html(fig, file=output_html, auto_open=False)
    except Exception as e:
        sys.exit(1)
    try:
        fig.show()
    except Exception as e:
        pass

def main():
    pose_graph_path = '/home/mrw9825/perception/pose_graph.json'
    images_file_path = '/home/mrw9825/perception/images.txt'
    pose_data = load_pose_graph(pose_graph_path)
    node_positions = {}
    for node_id, node_data in pose_data.items():
        t = node_data.get('t')
        if t and len(t) >= 2:
            x, y = t[0], t[1]
            node_positions[int(node_id)] = (x, y)
    image_positions, orientations = parse_images_file(images_file_path)
    combined_positions = {int(k): v for k, v in node_positions.items()}
    offset = max(combined_positions.keys()) + 1 if combined_positions else 0
    for image_id, pos in image_positions.items():
        new_id = offset + int(image_id)
        combined_positions[new_id] = pos
    distance_threshold = 30.0
    max_neighbors = 5
    edges_file = 'edges.txt'
    build_grid_map(combined_positions, distance_threshold=distance_threshold, max_neighbors=max_neighbors, edges_file=edges_file)
    graph_file = 'graph.pickle'
    graph, node_id_to_index = build_graph_from_edges(combined_positions, edges_file='edges.txt', graph_file=graph_file)
    start_index, end_index = set_start_end_nodes(graph, node_id_to_index, combined_positions)
    graph.write_pickle(graph_file)
    visualize_complete_graph(
        graph=graph,
        output_html='graph_visualization.html'
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(1)
