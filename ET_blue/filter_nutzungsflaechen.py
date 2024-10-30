import ee
from typing import Set, List


def get_crops_to_exclude() -> Set[str]:
    """
    Returns a set of crop types to exclude from irrigation analysis.

    Returns:
        Set[str]: A set of crop names to exclude.
    """
    return {
        "Andere Obstanlagen (Kiwis, Holunder usw.)",
        "Baumschule von Forstpflanzen ausserhalb der Forstzone",
        "Christbäume",
        "Hochstamm-Feldobstbäume (Punkte oder Flächen)",
        "Obstanlagen (Äpfel)",
        "Obstanlagen (Birnen)",
        "Obstanlagen (Steinobst)",
        "Obstanlagen aggregiert",
        "Reben",
        "Rebflächen mit natürlicher Artenvielfalt",
        "Übrige Baumschulen (Rosen, Früchte, usw.)",
        "Unbefestigte, natürliche Wege",
        "Wald",
        "Obstanlagen Steinobst",
        "Obstanlagen Äpfel",
        "Rebflächen mit natürlicher Artenvielfalt",
        "Hecken-, Feld und Ufergehölze (mit Krautsaum)",
        "Uferwiese (o.Wei.)  entlang von Fliessgew.",
        "Blühstreifen für Bestäuber und and. Nützlinge",
        "Saum auf Ackerfläche",
        "Landwirtschaftliche Produktion in Gebäuden (z. B. Champignon, Brüsseler)",
        "Ackerschonstreifen",
        "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen)",
        "Ruderalflächen, Steinhaufen und -wälle",
        "Uferwiese (ohne Weiden) entlang von Fließgewässern",
        "Wassergräben, Tümpel, Teiche",
        "Hecken-, Feld- und Ufergehölze (mit Puf.str.)",
        "Hecken, Feld-, Ufergehölze mit Krautsaum",
        "Saum auf Ackerflächen",
        "Uferwiesen (ohne Weiden) entlang von Fliessg.",
        "Wassergraben, Tümpel, Teiche",
        "Hecken-, Feld und Ufergehölz (reg. BFF)",
        "Hecken-, Feld- und Ufergehölze (mit Puf.str.)",
        "Uferwiese (ohne Weiden) entlang von Fliessgewässern",
        "Kunstwiese (ohne Weiden)",
        "Übr. Kunstwiese bb z.B. Schweine-, Geflügelwe.",
        "Übrige Kunstwiese (Schweine-, Geflügelweide)",
        "Kunstwiesen (ohne Weiden)",
        "Übrige Kunstwiese, beitragsberechtigt (z.B. Schweineweide, Geflügelweide)",
        "Üb. Grünfläche (Dauergrünfläche) beitragsbe.",
        "Übrige Dauerwiesen (ohne Weiden)",
        "Buntbrache",
        "Rotationsbrache",
        "Weide (Heimweiden, Üb. Weide ohne Sö.weiden)",
        "Wenig intensiv genutzte Wiesen (ohne Weiden)",
        "Extensiv genutzte Weiden",
        "Extensiv genutzte Wiesen (ohne Weiden)",
        "Üb. Grünfläche beitragsberechtigt",
        "Übrige Ackergewächse (nicht beitragsber.)",
        "Weide (Heimweiden, üb. Weide ohne Sö.geb.)",
        "Fläche ohne landw. Hauptzweckbestimmung",
        "Üb. Grünfläche nicht beitragsberechtigt",
        "Flächen ohne landwirtschaftliche Hauptzweckbestimmung (erschlossenes Bauland, Spiel-, Reit-, Camping-, Golf-, Flug- und Militärplätze oder ausgemarchte Bereiche von Eisenbahnen, öffentlichen Strassen und Gewässern)",
        "Übrige Grünfläche (Dauergrünfläche), beitragsberechtigt",
        "Übrige Grünfläche (Dauergrünfläche), nicht beitragsberechtigt",
        "übrige offene Ackerfläche, nicht beitragsberechtigt",
        "übrige Unproduktive Flächen (z.B. gemulchte Flächen, stark verunkraute Flächen, Hecke ohne Pufferstreifen",
        "Weiden (Heimweiden, übrige Weiden ohne Sömmerungsweiden)",
        "Wenig intensiv gen. Wiesen (ohne Weiden)",
        "Übrige Flächen innerhalb der LN, nicht beitragsberechtigt",
        "Blühstreifen für Bestäuber und andere Nützlinge",
        "Hecken-, Feld- und Ufergehölze (mit Pufferstreifen) (regionsspezifische Biodiversitätsförderfläche)",
        "Heuwiesen im Sömmerungsgebiet, Übrige Wiesen",
        "Mehrjährige nachwachsende Rohstoffe (Chinaschilf, usw.)",
        "Übrige Flächen mit Dauerkulturen, beitragsberechtigt",
        "Übrige Flächen mit Dauerkulturen, nicht beitragsberechtigt",
        "Übrige Grünfläche (Dauergrünflächen), nicht beitragsberechtigt",
        "Übrige unproduktive Flächen (z.B. gemulchte Flächen, stark verunkrautete Flächen, Hecken ohne Pufferstreifen)",
        "Uferwiesen entlang von Fliessgewässern (ohne Weiden)",
        "Ziersträucher, Ziergehölze und Zierstauden",
    }


def get_rainfed_reference_crops() -> set:
    """
    Returns a set of crop types to use as rainfed reference, excluding double-cropped fields.
    """
    return {"Kunstwiesen (ohne Weiden)", "Übrige Dauerwiesen (ohne Weiden)"}


def add_double_cropping_info(
    feature_collection: ee.FeatureCollection, double_cropping_image: ee.Image, scale=10
) -> ee.FeatureCollection:
    """
    Adds double cropping information to each feature based on the median value of pixels within the feature.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection of crop fields.
        double_cropping_image (ee.Image): Image with 'isDoubleCropping' band (1 for double-cropped, 0 for single-cropped).
        scale (int): The scale to use for reducing the image.

    Returns:
        ee.FeatureCollection: Updated feature collection with 'isDoubleCropped' property.
    """

    filled_image = double_cropping_image.unmask(0)

    def add_double_crop_property(feature):
        median_value = (
            filled_image.select("isDoubleCropping")
            .reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=feature.geometry(),
                scale=scale,
            )
            .get("isDoubleCropping")
        )

        return feature.set("isDoubleCropped", median_value)

    return feature_collection.map(add_double_crop_property)


def create_crop_filters(crops_to_exclude: set, rainfed_crops: set) -> tuple:
    """
    Creates filters for excluding crops and identifying rainfed reference crops.

    Args:
        crops_to_exclude (set): Set of crop names to exclude.
        rainfed_crops (set): Set of crop names to use as rainfed reference.

    Returns:
        tuple: A tuple containing two ee.Filter objects (exclude_condition, rainfed_condition).
    """
    exclude_condition = ee.Filter.inList("nutzung", list(crops_to_exclude)).Not()
    rainfed_condition = ee.Filter.And(
        ee.Filter.inList("nutzung", list(rainfed_crops)),
        ee.Filter.eq("isDoubleCropped", 0),  # Exclude double-cropped fields
    )
    return exclude_condition, rainfed_condition


def filter_crops(
    feature_collection: ee.FeatureCollection,
    exclude_filter: ee.Filter,
    rainfed_filter: ee.Filter,
) -> tuple:
    """
    Filters a feature collection based on crop type conditions.

    Args:
        feature_collection (ee.FeatureCollection): The input feature collection.
        exclude_filter (ee.Filter): Filter for excluding certain crop types.
        rainfed_filter (ee.Filter): Filter for identifying rainfed reference crops.

    Returns:
        tuple: A tuple containing two ee.FeatureCollection objects (filtered_fields, rainfed_fields).
    """
    filtered_fields = feature_collection.filter(exclude_filter)
    rainfed_fields = feature_collection.filter(rainfed_filter)
    return filtered_fields, rainfed_fields


def get_unique_nutzung(
    feature_collection: ee.FeatureCollection, nutzung_field_name: str = "nutzung"
) -> ee.List:
    """
    Gets all unique values for the 'nutzung' attribute in a FeatureCollection.

    Args:
        feature_collection (ee.FeatureCollection): The input FeatureCollection containing 'nutzung' property.
        nutzung_field_name (str): The name of the 'nutzung' field.

    Returns:
        ee.List: A list of unique 'nutzung' values.
    """
    return feature_collection.distinct("nutzung").aggregate_array("nutzung")


# Example usage
def main():
    # Load your feature collection and double cropping image
    nutzung_collection = ee.FeatureCollection("path/to/your/nutzung/collection")
    double_cropping_image = ee.Image("path/to/your/double_cropping_image")

    # Add double cropping information to the feature collection
    nutzung_collection_with_double_crop = add_double_cropping_info(
        nutzung_collection, double_cropping_image
    )

    crops_to_exclude = get_crops_to_exclude()
    rainfed_crops = get_rainfed_reference_crops()

    exclude_filter, rainfed_filter = create_crop_filters(
        crops_to_exclude, rainfed_crops
    )

    filtered_fields, rainfed_fields = filter_crops(
        nutzung_collection_with_double_crop, exclude_filter, rainfed_filter
    )

    print("Filtered fields count:", filtered_fields.size().getInfo())
    print("Rainfed reference fields count:", rainfed_fields.size().getInfo())


if __name__ == "__main__":
    main()
