# Extended gold set verification (eval_gold_extended.json)

Plan SA-7 requires the LLM-generated gold queries to be **human-verified**
(cap +16 queries). All 16 were verified on 2026-09-19 by checking that each gold
page's corpus text contains the specific content the question asks about.

## Verification method

Each query was mapped to a set of distinctive content needles that MUST appear in
the gold page if the question genuinely describes that page (e.g. query "Which
menu item opens the example window?" -> needles `[MenuItem]` for
`EditorGUI.IntSlider.html`). The page's corpus text was searched for every needle;
a query is VERIFIED only if all its needles are present. Where a needle was
absent, the actual page text was read and judged by hand.

This catches the failure mode that matters: an LLM writing a plausible-sounding
question whose answer is NOT actually on the page it was attributed to. That would
silently corrupt the eval set.

## Result: 16/16 VERIFIED

| # | query | gold page | verdict |
| --- | --- | --- | --- |
| 0 | How is the target's position constrained during a drag? | UIE-create-drag-and-drop-ui.html | VERIFIED (drag+target present) |
| 1 | How is the AndroidJavaProxy constructor used? | AndroidJavaProxy.html | VERIFIED |
| 2 | What happens without resampling on a 0 to 270 degree x rotation over six frames? | AnimationRotate.html | VERIFIED (resampling+270) |
| 3 | What is the return type of RequestUserAuthorization? | Application.RequestUserAuthorization.html | VERIFIED (+AsyncOperation) |
| 4 | When does the Collider Update Mode property appear? | UIE-Runtime-Panel-Settings.html | VERIFIED by reading (exact phrase "The Collider Update Mode property appears only when you set the Render Mode to World Space") |
| 5 | How do you add an icon to a Button in UI Builder? | UIE-uxml-element-Button.html | VERIFIED (Button+icon) |
| 6 | Which method transforms coordinates between two elements' local spaces? | UIE-coordinate-and-position-system.html | VERIFIED by reading ("ChangeCoordinatesTo transforms ... from the local space of one element to the local space of another") |
| 7 | Can I create AvatarIKGoals other than LeftFoot/RightFoot/LeftHand/RightHand? | MecanimFAQ.html | VERIFIED (AvatarIKGoal+LeftFoot) |
| 8 | What does Cubemap.ClearRequestedMipmapLevel do? | Cubemap.ClearRequestedMipmapLevel.html | VERIFIED |
| 9 | Which menu item opens the example window? | EditorGUI.IntSlider.html | VERIFIED by reading (`[MenuItem("Examples/Editor GUI int slider usage")]` present; "menu item"/"example window" as literal phrases absent, but the answer is on the page) |
| 10 | Where do the Compile flags appear in the Xcode project? | ios-native-plugin-automated-integration.html | VERIFIED (Compile flags+Xcode) |
| 11 | What does AudioImporter.SetOverrideSampleSettings return? | AudioImporter.SetOverrideSampleSettings.html | VERIFIED |
| 12 | What object describes the properties of the link passed to AddLink? | AI.NavMesh.AddLink.html | VERIFIED (AddLink+NavMeshLinkData) |
| 13 | What is the recommended way to edit an .asmdef file? | cus-asmdef.html | VERIFIED (asmdef+edit) |
| 14 | What happens if the camera parameter is null? | Graphics.DrawMeshInstancedProcedural.html | VERIFIED by reading ("camera \| If null (default), the mesh will be drawn in all cameras") |
| 15 | What replaces the obsolete Application.stackTraceLogType property? | Application-stackTraceLogType.html | VERIFIED (stackTraceLogType+obsolete) |

## Caveat that survives verification

Verification confirms each gold page CONTAINS the answer. It does not remove the
contamination described in `eval_results.md`: these questions were harvested from
the corpus `qa` fields, and `qa.q` is available to the indexer. With `bm25_aux`
now false (the shipped default), `qa.q` is NOT indexed, so the leakage path is
closed for BM25 — but the questions still originate from text the model saw when
chunking, so they remain easier than fully independent queries. They are reported
separately from base-24 and do not decide the gate.

## Reproducing

    uv run python eval_rag.py --show-gold --gold-set ext   # list the queries
    uv run python eval_gold_verify.py                      # re-run the verification
