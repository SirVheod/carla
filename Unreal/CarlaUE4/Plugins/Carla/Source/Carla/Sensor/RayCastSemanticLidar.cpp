// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// Copyright (c) 2021 FrostBit Software Lab
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include <PxScene.h>
#include <cmath>
#include "Carla.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Sensor/RayCastSemanticLidar.h"

#include <compiler/disable-ue4-macros.h>
#include "carla/geom/Math.h"
#include <compiler/enable-ue4-macros.h>

#include "DrawDebugHelpers.h"
#include "Engine/CollisionProfile.h"
#include "Runtime/Engine/Classes/Kismet/KismetMathLibrary.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"
#include <list>
#include <iostream>
#include <fstream>
using namespace std;

namespace crp = carla::rpc;

FActorDefinition ARayCastSemanticLidar::GetSensorDefinition()
{
  return UActorBlueprintFunctionLibrary::MakeLidarDefinition(TEXT("ray_cast_semantic"));
}

ARayCastSemanticLidar::ARayCastSemanticLidar(const FObjectInitializer& ObjectInitializer)
  : Super(ObjectInitializer)
{
  PrimaryActorTick.bCanEverTick = true;
}

void ARayCastSemanticLidar::Set(const FActorDescription &ActorDescription)
{
  Super::Set(ActorDescription);
  FLidarDescription LidarDescription;
  UActorBlueprintFunctionLibrary::SetLidar(ActorDescription, LidarDescription);
  Set(LidarDescription);
}

void ARayCastSemanticLidar::Set(const FLidarDescription &LidarDescription)
{
  Description = LidarDescription;
  SemanticLidarData = FSemanticLidarData(Description.Channels);
  CreateLasers();
  PointsPerChannel.resize(Description.Channels);
}

void ARayCastSemanticLidar::CreateLasers()
{
  const auto NumberOfLasers = Description.Channels;
  check(NumberOfLasers > 0u);
  const float DeltaAngle = NumberOfLasers == 1u ? 0.f :
    (Description.UpperFovLimit - Description.LowerFovLimit) /
    static_cast<float>(NumberOfLasers - 1);
  LaserAngles.Empty(NumberOfLasers);
  for(auto i = 0u; i < NumberOfLasers; ++i)
  {
    const float VerticalAngle =
        Description.UpperFovLimit - static_cast<float>(i) * DeltaAngle;
    LaserAngles.Emplace(VerticalAngle);
  }
}

void ARayCastSemanticLidar::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
  SimulateLidar(DeltaTime);

  auto DataStream = GetDataStream(*this);
  DataStream.Send(*this, SemanticLidarData, DataStream.PopBufferFromPool());
}

void ARayCastSemanticLidar::SimulateLidar(const float DeltaTime)
{
  const uint32 ChannelCount = Description.Channels;
  const uint32 PointsToScanWithOneLaser =
    FMath::RoundHalfFromZero(
        Description.PointsPerSecond * DeltaTime / float(ChannelCount));

  if (PointsToScanWithOneLaser <= 0)
  {
    UE_LOG(
        LogCarla,
        Warning,
        TEXT("%s: no points requested this frame, try increasing the number of points per second."),
        *GetName());
    return;
  }

  check(ChannelCount == LaserAngles.Num());

  const float CurrentHorizontalAngle = carla::geom::Math::ToDegrees(
      SemanticLidarData.GetHorizontalAngle());
  const float AngleDistanceOfTick = Description.RotationFrequency * Description.HorizontalFov 
      * DeltaTime;
  const float AngleDistanceOfLaserMeasure = AngleDistanceOfTick / PointsToScanWithOneLaser;

  ResetRecordedHits(ChannelCount, PointsToScanWithOneLaser);
  PreprocessRays(ChannelCount, PointsToScanWithOneLaser);

  auto *World = GetWorld();
  UCarlaGameInstance *GameInstance = UCarlaStatics::GetGameInstance(World);
  auto *Episode = GameInstance->GetCarlaEpisode();
  auto *Weather = Episode->GetWeather();
  FWeatherParameters w = Weather->GetCurrentWeather(); //current weather
  srand((unsigned)time( NULL )); //seed the random

  GetWorld()->GetPhysicsScene()->GetPxScene()->lockRead();
  ParallelFor(ChannelCount, [&](int32 idxChannel) {
    for (auto idxPtsOneLaser = 0u; idxPtsOneLaser < PointsToScanWithOneLaser; idxPtsOneLaser++) {
      FHitResult HitResult;
      const float VertAngle = LaserAngles[idxChannel];
      const float HorizAngle = std::fmod(CurrentHorizontalAngle + AngleDistanceOfLaserMeasure
          * idxPtsOneLaser, Description.HorizontalFov) - Description.HorizontalFov / 2;
      const bool PreprocessResult = RayPreprocessCondition[idxChannel][idxPtsOneLaser];

      if (PreprocessResult && ShootLaser(VertAngle, HorizAngle, HitResult, w)) {
        WritePointAsync(idxChannel, HitResult);
      }
    };
  });
  GetWorld()->GetPhysicsScene()->GetPxScene()->unlockRead();

  FTransform ActorTransf = GetTransform();
  ComputeAndSaveDetections(ActorTransf);

  const float HorizontalAngle = carla::geom::Math::ToRadians(std::fmod(CurrentHorizontalAngle + AngleDistanceOfTick, Description.HorizontalFov));
  SemanticLidarData.SetHorizontalAngle(HorizontalAngle);
}

void ARayCastSemanticLidar::ResetRecordedHits(uint32_t Channels, uint32_t MaxPointsPerChannel) {
  RecordedHits.resize(Channels);

  for (auto& hits : RecordedHits) {
    hits.clear();
    hits.reserve(MaxPointsPerChannel);
  }
}

void ARayCastSemanticLidar::PreprocessRays(uint32_t Channels, uint32_t MaxPointsPerChannel) {
  RayPreprocessCondition.resize(Channels);

  for (auto& conds : RayPreprocessCondition) {
    conds.clear();
    conds.resize(MaxPointsPerChannel);
    std::fill(conds.begin(), conds.end(), true);
  }
}

void ARayCastSemanticLidar::WritePointAsync(uint32_t channel, FHitResult &detection) {
  DEBUG_ASSERT(GetChannelCount() > channel);
  RecordedHits[channel].emplace_back(detection);
}

void ARayCastSemanticLidar::ComputeAndSaveDetections(const FTransform& SensorTransform) {
  for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel)
    PointsPerChannel[idxChannel] = RecordedHits[idxChannel].size();
  SemanticLidarData.ResetSerPoints(PointsPerChannel);

  for (auto idxChannel = 0u; idxChannel < Description.Channels; ++idxChannel) {
    for (auto& hit : RecordedHits[idxChannel]) {
      FSemanticDetection detection;
      ComputeRawDetection(hit, SensorTransform, detection);
      SemanticLidarData.WritePointSync(detection);
    }
  }
}

void ARayCastSemanticLidar::ComputeRawDetection(const FHitResult& HitInfo, const FTransform& SensorTransf, FSemanticDetection& Detection) const
{
    const FVector HitPoint = HitInfo.ImpactPoint;
    Detection.point = SensorTransf.Inverse().TransformPosition(HitPoint);

    const FVector VecInc = - (HitPoint - SensorTransf.GetLocation()).GetSafeNormal();
    Detection.cos_inc_angle = FVector::DotProduct(VecInc, HitInfo.ImpactNormal);

    const FActorRegistry &Registry = GetEpisode().GetActorRegistry();

    const AActor* actor = HitInfo.Actor.Get();
    Detection.object_idx = 0;

    
    if (HitInfo.Component == nullptr) 
    {
      Detection.object_tag = static_cast<uint32_t>(23);
    } else {
      Detection.object_tag = static_cast<uint32_t>(HitInfo.Component->CustomDepthStencilValue);
    }

    
    //Detection.object_tag = static_cast<uint32_t>(HitInfo.Component->CustomDepthStencilValue);

    if (actor != nullptr) {

      const FActorView view = Registry.Find(actor);
      if(view.IsValid())
        Detection.object_idx = view.GetActorId();

    }
    else {
      //UE_LOG(LogCarla, Warning, TEXT("Actor not valid %p!!!!"), actor);
    }
}

bool ARayCastSemanticLidar::CalculateNewHitPoint(FHitResult& HitInfo, float rain_amount, FVector end_trace, FVector LidarBodyLoc, FVector distance_to_hit) const
{
	FVector max_distance = distance_to_hit;																					// max_distance = point at 150m (lidar max range)
	FVector start_point = LidarBodyLoc;																						// start point is lidar position
	if (HitInfo.bBlockingHit)																								// If linetrace hits something
	{
		max_distance = HitInfo.ImpactPoint;																					// max_distance = where we got hit with linetrace
	}

	FVector vector = distance_to_hit - start_point; 
	FVector new_start_point = start_point + 0.01 * vector;																	// make start point away from center of lidar
	FVector new_vector = max_distance - new_start_point;																	// new vector from new start point to end point
	float random = (float) rand() / RAND_MAX;																				// random floating number between 0-1
	FVector new_hitpoint = new_start_point + random * new_vector;															// Generate new point from new start point to end point
	float distance = FVector::Dist(start_point, new_hitpoint) / 100;														// distance beteen new_hitpoint and start point (divide by 100 to get meters)
	
	float vis = 60 / (2000 - (rain_amount * 18.4f));
	float prob = vis*exp(-pow((distance-20.0f),2.0f)/pow(8.0f,2.0f))+vis*exp(-pow((distance-38.0f),2.0f)/pow(18.0f,2.0f));	// value between 0-1 this is the probability of trace to hit snowflake at certain distances

	float r = (float)rand() / RAND_MAX;																						// random between 0-1
	if (r < prob)																											// if random is smaller than probability from formula we hit the trace to snowflake
	{
		HitInfo.ImpactPoint = new_hitpoint;																					// assign new hitpoint
		return true;
	}
	else {
		return false;
	}
}

bool ARayCastSemanticLidar::ShootLaser(const float VerticalAngle, const float HorizontalAngle, FHitResult& HitResult, FWeatherParameters w) const
{
	FCollisionQueryParams TraceParams = FCollisionQueryParams(FName(TEXT("Laser_Trace")), true, this);
	TraceParams.bTraceComplex = true;
	TraceParams.bReturnPhysicalMaterial = false;

	FHitResult HitInfo(ForceInit);

	FTransform ActorTransf = GetTransform();
	FVector LidarBodyLoc = ActorTransf.GetLocation();
	FRotator LidarBodyRot = ActorTransf.Rotator();
	FRotator LaserRot (VerticalAngle, HorizontalAngle, 0);  // float InPitch, float InYaw, float InRoll
	FRotator ResultRot = UKismetMathLibrary::ComposeRotators(
		LaserRot,
		LidarBodyRot
	);
	const auto Range = Description.Range;
	FVector EndTrace = Range * UKismetMathLibrary::GetForwardVector(ResultRot) + LidarBodyLoc;
	FVector distance_to_hit = 15000 * UKismetMathLibrary::GetForwardVector(ResultRot) + LidarBodyLoc; //range is in cm

	GetWorld()->LineTraceSingleByChannel(
		HitInfo,
		LidarBodyLoc,
		EndTrace,
		ECC_GameTraceChannel2,
		TraceParams,
		FCollisionResponseParams::DefaultResponseParam
	);

	float temp = w.Temperature;
	float rain_amount = w.Precipitation;
	if (HitInfo.bBlockingHit) { 
		if (temp < 0 && rain_amount > 0) //If it is snowing
		{
			CalculateNewHitPoint(HitInfo, rain_amount, EndTrace, LidarBodyLoc, distance_to_hit);
		}
		HitResult = HitInfo; //equal to new hitpoint or the old one
		return true;
	} else { //If no hit is acquired
	if (temp < 0 && rain_amount > 0) //If it is snowing
	{
		if(CalculateNewHitPoint(HitInfo, rain_amount, EndTrace, LidarBodyLoc, distance_to_hit)) //if new hitpoint is made
		{
			HitResult = HitInfo;
			return true;
		}
	}
	return false;
	}
}
