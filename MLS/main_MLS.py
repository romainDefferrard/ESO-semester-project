import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch


class MLS:
    def __init__(self, filename, epsg, buffer):
        self.filename = filename
        self.epsg = epsg
        self.buffer = buffer

        warnings.filterwarnings("ignore", category=RuntimeWarning)

    def pipeline(self):
        self.open_file()
        self.add_buffer()
        self.intersections = self.get_intersections()
        self.add_times()
        #   self.most_inters()
        #   self.inters_sorted = self.intersections.sort_values(by="shared_length_m", ascending=True)

        self.intersections["n_segments"] = self.intersections.geometry.apply(lambda geom: self.gdf["buffer"].intersects(geom).sum())
        #  self.intersections = self.intersections.sort_values(by="n_segments", ascending=False)

        # print(self.intersections[:5])

        zones_ids = self.extract_most_visited(top_n=5, plot=True)
        zone_key = list(zones_ids.keys())[0]
        zone_entries = zones_ids.get(zone_key)
        ids = [entry["id"] for entry in zone_entries]
        print(ids)

    def extract_most_visited(self, top_n=5, plot=True, margin=100):

        sorted_zones = self.intersections.sort_values(by="n_segments", ascending=False).head(top_n)
        print(sorted_zones[:1])
        zone_id_map = {}

        for i, row in sorted_zones.iterrows():
            zone_geom = row["overlap_geom"]

            # Find segments intersecting this zone
            overlapping_segs = self.gdf[self.gdf["buffer"].intersects(zone_geom)]
            segment_info = overlapping_segs[["id", "starttime", "endtime"]].to_dict(orient="records")
            zone_id_map[f"zone_{i}"] = segment_info

            if plot:
                fig, ax = plt.subplots(figsize=(8, 6))

                # Create a unique color per ID using a colormap
                ids = overlapping_segs["id"].unique()
                num_ids = len(ids)
                legend_handles = []
                base_cmap = plt.colormaps["tab20"]
                colors = [base_cmap(i / num_ids) for i in range(num_ids)]

                # Plot each buffer with its own color
                for idx, seg_id in enumerate(ids):
                    seg = overlapping_segs[overlapping_segs["id"] == seg_id]
                    base_color = colors[idx]

                    fill_color = to_rgba(base_color, alpha=0.3)
                    edge_color = to_rgba(base_color, alpha=0.9)

                    seg["buffer"].plot(ax=ax, color=fill_color, edgecolor=edge_color, linewidth=1.2)
                    seg["geometry"].plot(ax=ax, color=fill_color, edgecolor=edge_color, linewidth=1.2)

                    legend_handles.append(Patch(facecolor=fill_color, edgecolor=edge_color, label=f"id {seg_id}"))

                # Plot the intersection geometry
                gpd.GeoSeries(zone_geom).plot(ax=ax, color="none", edgecolor="red", linewidth=2, label=f"Zone {i}")

                # Zoom to the intersection
                minx, miny, maxx, maxy = zone_geom.bounds
                ax.set_xlim(minx - margin, maxx + margin)
                ax.set_ylim(miny - margin, maxy + margin)

                ax.set_title(f"Zone {i} — {len(ids)} segments")
                ax.set_aspect("equal")

                ax.legend(handles=legend_handles, loc="upper right", fontsize="small")

                plt.tight_layout()
                plt.show()

        return zone_id_map

    def open_file(self):
        self.gdf = gpd.read_file(self.filename)
        self.gdf = self.gdf.to_crs(epsg=self.epsg)

    def add_buffer(self):
        self.gdf["buffer"] = self.gdf.geometry.buffer(10)

    def get_intersections(self):

        records = []

        for i in range(len(self.gdf)):
            for j in range(i + 1, len(self.gdf)):
                buf1 = self.gdf.loc[i, "buffer"]
                buf2 = self.gdf.loc[j, "buffer"]

                if buf1.intersects(buf2):
                    overlap_poly = buf1.intersection(buf2)

                    if not overlap_poly.is_empty:
                        # Intersect original lines with the overlap polygon
                        line1 = self.gdf.loc[i, "geometry"].intersection(overlap_poly)
                        line2 = self.gdf.loc[j, "geometry"].intersection(overlap_poly)

                        # Take union of both clipped lines (they may be on opposite sides of the polygon)
                        shared_line = line1.union(line2)

                        # Get total length of shared line(s)
                        shared_length = shared_line.length if not shared_line.is_empty else 0

                        records.append(
                            {
                                "id_1": self.gdf.loc[i, "id"],
                                "id_2": self.gdf.loc[j, "id"],
                                "overlap_geom": overlap_poly,
                                "shared_length_m": shared_length,
                                "overlap_area_m2": overlap_poly.area,
                            }
                        )
        intersections = gpd.GeoDataFrame(records, geometry="overlap_geom", crs=self.gdf.crs)
        return intersections

    def add_times(self):
        gdf_indexed = self.gdf.set_index("id")

        self.intersections["t1_start"] = self.intersections["id_1"].map(gdf_indexed["starttime"])
        self.intersections["t1_end"] = self.intersections["id_1"].map(gdf_indexed["endtime"])

        self.intersections["t2_start"] = self.intersections["id_2"].map(gdf_indexed["starttime"])
        self.intersections["t2_end"] = self.intersections["id_2"].map(gdf_indexed["endtime"])
        self.intersections.reset_index(drop=True)

    def most_inters(self):
        new_gdf = self.intersections["n_segments"] = self.intersections.geometry.apply(lambda geom: self.intersections["buffer"].intersects(geom).sum())
        return new_gdf


if __name__ == "__main__":
    filename = "Data/EXTEND_SDC_LEFT_M230905.shp"
    epsg = 2056
    buffer = 10

    mls = MLS(filename, epsg, buffer)
    mls.pipeline()


# %%
