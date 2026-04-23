from mantis_agent.extraction import ExtractionResult


def test_private_seller_is_viable():
    result = ExtractionResult(
        year="2006",
        make="Luhrs",
        model="41 Convertible",
        price="$235,000",
        phone="+507 6615-9404",
        url="boattrader.com/boat/2006-luhrs-41-convertible-10131961/",
        seller="Mario Vega",
    )

    assert result.is_private_seller()
    assert result.is_viable()


def test_dealer_inventory_is_not_viable_even_with_phone():
    result = ExtractionResult(
        year="2026",
        make="Azimut",
        model="S8",
        price="Request a Price",
        phone="954-800-6512",
        url="boattrader.com/boat/2026-azimut-s8-10041625/",
        seller="MarineMax East Florida Yacht Center",
    )

    assert not result.is_private_seller()
    assert not result.is_viable()


def test_dealer_url_is_not_viable():
    result = ExtractionResult(
        year="2026",
        make="Azimut",
        model="Verve 48",
        price="Request a Price",
        phone="",
        url="boattrader.com/boats/dealerName-MarineMax/make-azimut/condition-new/",
    )

    assert not result.is_private_seller()
    assert not result.is_viable()
