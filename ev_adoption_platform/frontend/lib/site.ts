export const siteLinks = {
  github: "https://github.com/Atharva2102/ev-buyer-intelligence",
  linkedin: "https://www.linkedin.com/"
} as const;

const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const publicBasePath = configuredBasePath === "/"
  ? ""
  : configuredBasePath.replace(/\/$/, "");
export const publicAsset = (path: string) => `${publicBasePath}${path}`;

export const vehicleImages: Record<string, string> = {
  Hatchback: publicAsset("/images/vehicles/hatchback.png"),
  Sedan: publicAsset("/images/vehicles/sedan.png"),
  SUV: publicAsset("/images/vehicles/suv.png"),
  Truck: publicAsset("/images/vehicles/truck.png")
};
