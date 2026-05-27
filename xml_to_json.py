import os
import glob
import json
import xml.etree.ElementTree as ET

# --- STEP 1: Convert XML batches to clean JSONs ---
def convert_xml_to_json(xml_file, json_file):
    results = []
    try:
        context = ET.iterparse(xml_file, events=("end",))
        for event, elem in context:
            if elem.tag.endswith('page'):
                title = elem.find('./{*}title')
                revision = elem.find('./{*}revision')
                text = revision.find('./{*}text') if revision is not None else None
                if title is not None and text is not None and text.text:
                    results.append({"title": title.text, "text": text.text})
                elem.clear()
    except ET.ParseError as e:
        print(f"Parse error in {xml_file}: {e}. Skipping this file.")
        return  # Skip this file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {json_file}")


# --- STEP 2: Merge all JSONs for each anime ---
def merge_same_anime_jsons(wiki_name):
    batch_jsons = sorted(glob.glob(os.path.join("json_files", f"{wiki_name}_articles_*.json")))
    merged = []
    for json_file in batch_jsons:
        with open(json_file, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    merged_file = os.path.join("merged_json_files", f"{wiki_name}_all_articles_merged.json")
    with open(merged_file, "w", encoding="utf-8") as out_f:
        json.dump(merged, out_f, ensure_ascii=False, indent=2)
    print(f"Merged {wiki_name} into {merged_file}")


# --- STEP 3: Merge all anime into a single JSON ---
def merge_all_jsons():
    merged = []
    for f in glob.glob(os.path.join("merged_json_files", "*_all_articles_merged.json")):
        with open(f, "r", encoding="utf-8") as infile:
            merged.extend(json.load(infile))
    with open("all_anime_articles.json", "w", encoding="utf-8") as outfile:
        json.dump(merged, outfile, ensure_ascii=False, indent=2)
    print(f"All anime merged into {"all_anime_articles.json"}")



if __name__ == "__main__":
    # Step 1: Convert all XML batches to clean JSONs
    xml_files = glob.glob(os.path.join("xml_files", "*.xml"))
    for xml_file in xml_files:
        json_file = os.path.join("json_files", os.path.basename(xml_file).replace(".xml", ".json"))
        convert_xml_to_json(xml_file, json_file)

    # Step 2: Merge all JSONs for each anime
    # List your anime wiki names (should match the prefix in your XML/JSON filenames)
    wiki_names = [
        "naruto", "kimetsu-no-yaiba", "codegeass", "onepiece", "haikyuu", "deathnote", "fullmetalalchemist",
        "gintama", "onepunchman", "jojowiki", "berserk", "blackclover", "bleach", "dragonball", "maid-sama",
        "fruitsbasket", "myheroacademia", "saikikusuo", "evangelion", "madeinabyss", "kill-la-kill",
        "jujutsukaisen", "chainsawman", "yourlieinapril"
    ]
    for wiki_name in wiki_names:
        merge_same_anime_jsons(wiki_name)

    # Step 3: Merge all anime into a single JSON
    merge_all_jsons()
