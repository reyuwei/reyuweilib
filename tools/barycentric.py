import torch


# # compute bary centric coordinate
# triangles = target_mesh.triangles
# mesh_query = trimesh.proximity.ProximityQuery(target_mesh)
# closest, distance, triangle_id = mesh_query.on_surface(target_lmk)
# bmc = trimesh.triangles.points_to_barycentric(triangles[triangle_id], target_lmk)

def vertices2landmarks(
    vertices,
    faces,
    lmk_faces_idx,
    lmk_bary_coords
):
    ''' Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        # batch_size, -1, 3)
    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        1, -1, 3)
    lmk_faces = lmk_faces.repeat([batch_size,1,1])

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)
    landmarks = torch.einsum('blfi,lf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks
