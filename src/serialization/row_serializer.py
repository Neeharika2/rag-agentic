import re
from typing import List, Dict, Any
from src.serialization.serializer_interface import SerializerInterface

class RowSerializer(SerializerInterface):
    """
    Concrete implementation of SerializerInterface.
    Responsible for serializing text paragraphs and table rows into 
    standardized vector store chunks with dynamic metadata extraction.
    """

    def classify_section(self, section_name: str) -> str:
        """Dynamically maps PDF headers to section codes."""
        s = str(section_name).lower()
        if "team profile" in s or "section 1" in s:
            return "team"
        elif "batting" in s or "section 2" in s:
            return "batting"
        elif "bowling" in s or "section 3" in s:
            return "bowling"
        elif "head-to-head" in s or "head to head" in s or "section 4" in s:
            return "h2h"
        elif "venue" in s or "pitch" in s or "section 5" in s:
            return "venue"
        elif "season" in s or "performance" in s or "section 6" in s:
            return "season"
        elif "recent form" in s or "section 7" in s:
            return "form"
        elif "records" in s or "milestone" in s or "section 8" in s:
            return "records"
        elif "conflicting" in s or "noisy" in s or "section 11" in s:
            return "validation"
        return "general"

    def extract_metadata(self, row: Dict[str, str], section_code: str) -> Dict[str, Any]:
        """Dynamically extracts metadata keys from row attributes without hardcoding."""
        metadata = {"section": section_code}
        
        for key, val in row.items():
            k_lower = key.lower()
            val_strip = str(val).strip()
            
            # Extract player name
            if "player" in k_lower or "batsman" in k_lower or "bowler" in k_lower:
                metadata["player_name"] = val_strip
            
            # Extract team names (including Team 1 / Team 2 or Short)
            if "team" in k_lower or "short" in k_lower:
                if "team 1" in k_lower or "team1" in k_lower:
                    metadata["team1"] = val_strip
                elif "team 2" in k_lower or "team2" in k_lower:
                    metadata["team2"] = val_strip
                else:
                    metadata["team"] = val_strip
            
            # Extract role
            if "role" in k_lower:
                metadata["role"] = val_strip
                
            # Extract bowl type
            if "type" in k_lower or "bowl_type" in k_lower or "bowling style" in k_lower:
                metadata["bowl_type"] = val_strip
                
            # Extract pitch type
            if "pitch" in k_lower:
                metadata["pitch_type"] = val_strip
                
            # Extract venue names
            if "venue" in k_lower or "stadium" in k_lower or "ground" in k_lower:
                metadata["venue_name"] = val_strip
            
            # Extract category (for records)
            if "category" in k_lower or "milestone" in k_lower:
                metadata["category"] = val_strip

        # Section specific default seasons
        if section_code in ["team", "form"]:
            metadata["season"] = 2024
            
        return metadata

    def serialize(self, parsed_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        
        for element in parsed_elements:
            section_raw = element.get("section", "general")
            section_code = self.classify_section(section_raw)
            el_type = element.get("type")
            
            if el_type == "text":
                text_val = element.get("text", "")
                chunks.append({
                    "text": f"{text_val}. Section: {section_code}.",
                    "metadata": {"section": section_code, "type": "narrative"}
                })
                
            elif el_type == "table":
                rows = element.get("data", [])
                headers = element.get("headers", [])
                
                # Check for special table structures dynamically:
                
                # Case A: Validation / Conflicting Data (Section 11)
                if section_code == "validation":
                    for row in rows:
                        # Find primary/secondary value headers dynamically
                        player_fact = next((row[k] for k in row if "player" in k.lower() or "fact" in k.lower()), "Unknown")
                        conflict_type = next((row[k] for k in row if "conflict" in k.lower() or "type" in k.lower()), "Discrepancy")
                        
                        primary_header = next((h for h in headers if "primary" in h.lower()), None)
                        secondary_header = next((h for h in headers if "secondary" in h.lower()), None)
                        
                        if primary_header and secondary_header:
                            prim_val = row[primary_header]
                            sec_val = row[secondary_header]
                            
                            # Emit primary chunk
                            chunks.append({
                                "text": f"Entity: {player_fact}, Metric: {conflict_type}, Value: {prim_val}. Source: primary. Section: validation.",
                                "metadata": {
                                    "section": "validation",
                                    "player_name": player_fact,
                                    "metric": conflict_type,
                                    "source": "primary",
                                    "conflict": True
                                }
                            })
                            
                            # Emit secondary chunk
                            chunks.append({
                                "text": f"Entity: {player_fact}, Metric: {conflict_type}, Value: {sec_val}. Source: secondary. Section: validation.",
                                "metadata": {
                                    "section": "validation",
                                    "player_name": player_fact,
                                    "metric": conflict_type,
                                    "source": "secondary",
                                    "conflict": True
                                }
                            })
                    continue
                
                # Case B: Season-wise Results (Section 6) - 1 row per team per year
                year_headers = [h for h in headers if re.match(r"^\d{4}", h)]
                if section_code == "season" and year_headers:
                    for row in rows:
                        team_val = next((row[k] for k in row if "team" in k.lower() or "short" in k.lower()), "Unknown")
                        for yr_h in year_headers:
                            year_int = int(re.match(r"^(\d{4})", yr_h).group(1))
                            pos_val = row[yr_h]
                            chunks.append({
                                "text": f"Team: {team_val}, Year: {year_int}, Position: {pos_val}. Section: season.",
                                "metadata": {
                                    "section": "season",
                                    "team": team_val,
                                    "year": year_int
                                }
                            })
                    continue
                
                # Standard Tabular Serialization (1 row = 1 chunk)
                for row in rows:
                    meta = self.extract_metadata(row, section_code)
                    
                    # Bidirectional H2H index helper
                    if section_code == "h2h":
                        t1 = meta.get("team1")
                        t2 = meta.get("team2")
                        if t1 and t2:
                            meta["teams"] = [t1, t2] # list format for contains lookup
                    
                    # Generate key-value serialized string
                    kv_pairs = []
                    for key, val in row.items():
                        kv_pairs.append(f"{key}: {val}")
                    serialized_text = ", ".join(kv_pairs) + f". Section: {section_code}."
                    
                    chunks.append({
                        "text": serialized_text,
                        "metadata": meta
                    })
                    
        return chunks
