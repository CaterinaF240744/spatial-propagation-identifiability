"""
build_reference.py -- convert the GSE107585 mouse kidney single-cell matrix (Park et al., Science 2018)
into the AnnData reference used for label transfer (obs['cell_type'] = original cluster number).

    python preprocessing/build_reference.py <Mouse_kidney_single_cell_datamatrix.txt> data/GSE107585_sc.h5ad
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import scanpy as sc


def convert_sc_to_h5ad(
    infile:      str = "Mouse_kidney_single_cell_datamatrix.txt",
    outfile:     str = "GSE107585_sc.h5ad",
    cluster_row: str = "Cluster_Number",
    dtype=np.float32,
) -> sc.AnnData:
    """
    Convert the GSE107585 counts matrix (.txt, transposed format) to AnnData.

    Handles rows with variable token counts: pads with zeros if missing,
    truncates if in excess. The ``Cluster_Number`` row is extracted as a
    cell label and excluded from the count matrix.

    Parameters
    ----------
    infile      : path to the input .txt file
    outfile     : output path for the .h5ad file
    cluster_row : name of the row containing cell cluster labels
    dtype       : numpy dtype for counts (default float32)

    Returns
    -------
    adata : AnnData (cells × genes) with ``obs['cell_type']``
    """
    with open(infile, "r") as f:
        header   = f.readline().rstrip("\n").split("\t")
        cell_ids = header[1:]
        if cell_ids and cell_ids[-1] == "":
            cell_ids = cell_ids[:-1]
        n_cells = len(cell_ids)
        print(f"[convert_sc_to_h5ad] detected cells: {n_cells}")

        data, indices, indptr = [], [], [0]
        genes      = []
        cell_types = None

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            gene, rest = line.split("\t", 1)
            toks = rest.split("\t")
            if toks and toks[-1] == "":
                toks = toks[:-1]

            # Normalise token length
            if len(toks) > n_cells:
                toks = toks[:n_cells]
            elif len(toks) < n_cells:
                toks = toks + ["0"] * (n_cells - len(toks))

            if gene == cluster_row:
                cell_types = np.array(toks, dtype=str)
                continue

            arr = np.asarray(toks, dtype=dtype)
            nz  = np.nonzero(arr)[0]
            if nz.size:
                data.extend(arr[nz].tolist())
                indices.extend(nz.tolist())
            indptr.append(len(data))
            genes.append(gene)

    if cell_types is None:
        raise ValueError(f"Row '{cluster_row}' not found in {infile}.")

    X_sp = sp.csr_matrix(
        (np.array(data,    dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr,  dtype=np.int64)),
        shape=(len(genes), n_cells),
    )
    adata = sc.AnnData(X=X_sp.T)
    adata.obs_names = cell_ids
    adata.var_names = genes
    adata.obs["cell_type"] = cell_types
    adata.write_h5ad(outfile, compression="gzip")
    print(f"[convert_sc_to_h5ad] saved to: {os.path.abspath(outfile)}")
    return adata


if __name__ == "__main__":
    convert_sc_to_h5ad(sys.argv[1], outfile=sys.argv[2])
