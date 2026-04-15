# BoatTrader → PopYachts Lead Entry Workflow Specification

## Inputs
- `search_radius`: distance in miles from zip code
- `zip_code`: US zip code for search center  
- `pop_password`: password for PopYachts admin authentication

## Interaction Model
Execute through normal browser interactions only. Mouse clicks, keyboard typing, scrolling, reading visible text. No developer tools, no console, no scripts, no automation frameworks.

## Viability Check Rules
A listing is VIABLE only if a phone number exists in Description or More Details.

Phone patterns: (555) 555-5555, 555-555-5555, 555.555.5555, 5555555555, +1 555-555-5555, international formats with 7+ digits.

NOT viable: partial/obfuscated numbers, <7 digits, prices/years/zips.

Extract FIRST valid phone number + seller name (or "Unknown").

## Lead Data Fields
- year: 4-digit year from title/details
- make: manufacturer from title/details
- model: model name from title/details
- boat_type: from More Details or breadcrumbs
- asking_price: numeric only, no $ or commas
- listing_url: from browser address bar
- seller_name: from viability check
- seller_phone: from viability check

## Duplicate Detection
Check by phone (digits only) or listing_url against previously processed leads.

## PopYachts Authentication
1. Navigate to https://admin.popyachts.com/ (root, NOT subpages)
2. Sign in with Google: svsg@popsells.com / {pop_password}
3. If 2FA requested, use TOTP tool
4. ONE login attempt only — do not retry on failure
5. Scroll DOWN past dashboard to find "User Management" section
6. Select "Erin NesSmith" (NOT the top-left dropdown)
7. Then navigate to lead entry: https://admin.popyachts.com/Page/Account-Main-Seller-Lead-Review-External/NA/Page.html

## Lead Entry
- Select "Outside Boat Lead"
- Fill: year, make, model, boat_type, asking_price, seller_name, seller_phone, listing_url
- Submit and check URL change for success
- Silent rejection = form clears but URL unchanged — do NOT retry

## Tab Management
- Tab 1: BoatTrader results (always open)
- Tab 2: PopYachts form (open on first viable lead, keep open)
- Tab 3: Individual listing (temporary, close after processing)

## Pagination
Process all listings on page, then click Next. listing_index continues across pages.

## Stopping
Stop when no more pages. Generate completion report with all leads processed.
